import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class VisualHead(torch.nn.Module):
    def __init__(self, 
        cls_num, input_size=768, hidden_size=512, ff_size=2048, pe=False, head_drop_rate=0.,
        ff_kernelsize=3, pretrained_ckpt=None, is_empty=False, frozen=False, 
        plus_conv_cfg={},
        ssl_projection_cfg={}):
        super().__init__()
        self.is_empty = is_empty
        self.plus_conv_cfg = plus_conv_cfg
        self.ssl_projection_cfg = ssl_projection_cfg
        if is_empty==False:
            self.frozen = frozen
            self.hidden_size = hidden_size

            if input_size is None:
                self.fc1 = nn.Identity()
            else:
                self.fc1 = torch.nn.Linear(input_size, self.hidden_size)
            
            
            
            
            self.bn1 = MaskedNorm(num_features=self.hidden_size, norm_type='sync_batch')
            # self.bn1 = nn.SyncBatchNorm(num_features=self.hidden_size)



            
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(p=head_drop_rate)

            if pe:
                self.pe = PositionalEncoding(self.hidden_size)
            else:
                self.pe = torch.nn.Identity()

            self.feedforward = PositionwiseFeedForward(input_size=self.hidden_size,
                ff_size=ff_size,
                dropout=head_drop_rate, kernel_size=ff_kernelsize, skip_connection=True)
            
            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)

            if plus_conv_cfg!={}:
                plus_convs = []
                for i in range(plus_conv_cfg['num_layer']):
                    plus_convs.append(nn.Conv1d(self.hidden_size, self.hidden_size, 
                        kernel_size=plus_conv_cfg['kernel_size'], stride=plus_conv_cfg['stride'], padding_mode='replicate'))
                self.plus_conv = nn.Sequential(*plus_convs)
            else:
                self.plus_conv = nn.Identity()

            if ssl_projection_cfg!={}:
                self.ssl_projection = MLPHead(embedding_size=self.hidden_size, 
                    projection_hidden_size=ssl_projection_cfg['hidden_size'])

            self.gloss_output_layer = torch.nn.Linear(self.hidden_size, cls_num)

            if self.frozen:
                self.frozen_layers = [self.fc1, self.bn1, self.relu1,  self.pe, self.dropout1, self.feedforward, self.layer_norm]
                for layer in self.frozen_layers:
                    for name, param in layer.named_parameters():
                        param.requires_grad = False
                    layer.eval()
        else:
            self.gloss_output_layer = torch.nn.Linear(input_size, cls_num)
        if pretrained_ckpt:
            self.load_from_pretrained_ckpt(pretrained_ckpt)

    # def load_from_pretrained_ckpt(self, pretrained_ckpt):
    #     logger = get_logger()
    #     checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
    #     load_dict = {}
    #     for k,v in checkpoint.items():
    #         if 'recognition_network.visual_head.' in k:
    #             load_dict[k.replace('recognition_network.visual_head.','')] = v
    #     self.load_state_dict(load_dict)
    #     logger.info('Load Visual Head from pretrained ckpt {}'.format(pretrained_ckpt))

    def forward(self, x, mask, valid_len_in=None):
        B, Tin, D = x.shape 
        if self.is_empty==False:
            if not self.frozen:
                #projection 1
                x = self.fc1(x)
                x = self.bn1(x, mask)

                # Suppose x has shape [B, T, hidden_size]:
                # x = x.transpose(1, 2)  # now [B, hidden_size, T]
                # x = self.bn1(x)        # standard SyncBatchNorm call
                # x = x.transpose(1, 2)  # back to [B, T, hidden_size]


                x = self.relu1(x)
                #pe
                x = self.pe(x)
                x = self.dropout1(x)

                #feedforward
                x = self.feedforward(x)
                x = self.layer_norm(x)

                x = x.transpose(1,2)
                x = self.plus_conv(x)
                x = x.transpose(1,2)
            else:
                with torch.no_grad():
                    for ii, layer in enumerate(self.frozen_layers):
                        layer.eval()
                        if ii==1:
                            x = layer(x, mask)
                        else:
                            x = layer(x)
                x = x.transpose(1,2)
                x = self.plus_conv(x)
                x = x.transpose(1,2)

        #classification
        logits = self.gloss_output_layer(x) #B,T,V
        gloss_probabilities_log = logits.log_softmax(2) 
        gloss_probabilities = logits.softmax(2)

        if self.plus_conv_cfg!={}:
            B, Tout, D = x.shape
            valid_len_out = torch.floor(valid_len_in*Tout/Tin).long() #B,
        else:
            valid_len_out = valid_len_in
        if self.ssl_projection_cfg!={}:
            x_ssl = self.ssl_projection(x)
            if self.ssl_projection_cfg['normalize']==True:
                x_ssl = F.normalize(x_ssl, dim=-1)
        else:
            x_ssl = None
            
        return {
            # 'gloss_feature_ssl':x_ssl, 
            # 'gloss_feature': x,
            # 'gloss_feature_norm': F.normalize(x, dim=-1),
            'gloss_logits':logits, 
            'gloss_probabilities_log':gloss_probabilities_log,
            # 'gloss_probabilities': gloss_probabilities,
            'valid_len_out':valid_len_out
        }

class MLPHead(nn.Module):
    def __init__(self, embedding_size, projection_hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(nn.Linear(self.embedding_size, projection_hidden_size),
                                nn.BatchNorm1d(projection_hidden_size),
                                nn.ReLU(True),
                                nn.Linear(projection_hidden_size, self.embedding_size))
    
    def forward(self, x):
        b, l, c = x.shape
        x = x.reshape(-1,c)#x.view(-1,c)
        x = self.net(x)
        return x.reshape(b,l,c)#x.view(b, l, c)

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1, kernel_size=1,
        skip_connection=True):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.kernel_size = kernel_size
        if type(self.kernel_size)==int:
            conv_1 = nn.Conv1d(input_size, ff_size, kernel_size=kernel_size, stride=1, padding='same')
            conv_2 = nn.Conv1d(ff_size, input_size, kernel_size=kernel_size, stride=1, padding='same')
            self.pwff_layer = nn.Sequential(
                conv_1,
                nn.ReLU(),
                nn.Dropout(dropout),
                conv_2,
                nn.Dropout(dropout),
            )
        elif type(self.kernel_size)==list:
            pwff = []
            first_conv = nn.Conv1d(input_size, ff_size, kernel_size=kernel_size[0], stride=1, padding='same')
            pwff += [first_conv, nn.ReLU(), nn.Dropout(dropout)]
            for ks in kernel_size[1:-1]:
                conv = nn.Conv1d(ff_size, ff_size, kernel_size=ks, stride=1, padding='same')
                pwff += [conv, nn.ReLU(), nn.Dropout(dropout)]
            last_conv = nn.Conv1d(ff_size, input_size, kernel_size=kernel_size[-1], stride=1, padding='same')
            pwff += [last_conv, nn.Dropout(dropout)]

            self.pwff_layer = nn.Sequential(
                *pwff
            )
        else:
            raise ValueError
        self.skip_connection=skip_connection
        if not skip_connection:
            print('Turn off skip_connection in PositionwiseFeedForward')

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_t = x_norm.transpose(1,2)
        x_t = self.pwff_layer(x_t)
        if self.skip_connection:
            return x_t.transpose(1,2)+x
        else:
            return x_t.transpose(1,2)

class MaskedNorm(nn.Module):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, num_features=512, norm_type='sync_batch', num_groups=1):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            raise ValueError("Please use sync_batch")
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == 'sync_batch':
            self.norm = nn.SyncBatchNorm(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )

            # print("MaskedNorm: selected elements =", selected.numel())


            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])

class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]
    


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a dummy VisualHead instance
    cls_num = 1235
    input_size = 768
    hidden_size = 512
    head = VisualHead(cls_num, input_size=input_size, hidden_size=hidden_size, ff_size=2048, pe=False, head_drop_rate=0.1)
    head.to(device)

    # Create dummy input: assume batch size 2, T=32, and feature dimension 768
    dummy_x = torch.randn(2, 32, input_size, requires_grad=True).to(device)
    # Create a dummy mask of ones (or you can experiment with a realistic mask)
    dummy_mask = torch.ones(2, 1, 32, dtype=torch.bool).to(device)
    # Create a dummy valid length tensor
    valid_len = torch.tensor([32, 32]).to(device)

    # Forward pass through the head
    out = head(dummy_x, dummy_mask, valid_len)
    # Use gloss_logits to compute a simple scalar loss
    loss = out['gloss_logits'].sum()
    loss.backward()

    # Check a few parameters for gradients:
    for name, param in head.named_parameters():
        if param.requires_grad:
            print(f"{name}: grad norm = {param.grad.norm() if param.grad is not None else 'None'}")
