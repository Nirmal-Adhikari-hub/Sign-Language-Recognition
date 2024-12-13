import torch
import gzip, pickle


if __name__ == "__main__":
    anno_path="/nas/Dataset/Phoenix/phoenix-2014.test"
    gloss_to_id_path="/nas/Dataset/Phoenix/gloss2ids.pkl"
    video_path="/nas/Dataset/Phoenix/phoenix-2014-videos.zip"


    with open(gloss_to_id_path, 'rb') as f:
        gloss_to_id = pickle.load(f)

    with gzip.open(anno_path) as f:
        annotations = pickle.load(f)

    # i = 0
    # indices = []
    # for idx, anno in enumerate(annotations): 
    #         gloss = anno['gloss']
    #         words = gloss.split()
    #         gloss_ids = [gloss_to_id[word] for word in words]
    #         if 'MEIN' in words:
    #             # print(f"{gloss}")
    #             i += 1
    #             indices.append(idx)
    # print(f"MEIN appeared {i} times in {indices}'th indices.")
    # print(annotations[0])
    print(gloss_to_id)