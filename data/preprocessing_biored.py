import json
from tqdm import tqdm
import numpy as np
bio_rel2id = {'Na':0, 'Association': 1, 'Positive_Correlation': 2, 'Negative_Correlation': 3, 'Bind': 4, 'Drug_Interaction': 5, 'Cotreatment': 6, 'Comparison': 7, 'Conversion': 8}

def adjust_mention_positions(mention_pos, sents):
    """
    Điều chỉnh men_start và men_end trong mention_pos bằng cách trừ đi tổng số từ của các câu trước đó.

    Args:
        mention_pos (list): Danh sách các tuple mention_pos.
                            Mỗi tuple có dạng (men_start, men_end, ent_id, men_sen_id, men_index, men_index+entity_number).
        sents (list): Danh sách các câu, mỗi câu là một danh sách các từ.

    Returns:
        list: mention_pos đã được điều chỉnh.
    """
    adjusted_mention_pos = []
    
    # Tính tổng độ dài (số từ) của các câu trước mỗi câu
    sentence_lengths = [len(s) for s in sents]
    
    # Tạo một list chứa tổng độ dài tích lũy của các câu trước đó
    # Ví dụ: [0, len(sents[0]), len(sents[0]) + len(sents[1]), ...]
    cumulative_lengths = [0] * len(sents)
    for i in range(1, len(sents)):
        cumulative_lengths[i] = cumulative_lengths[i-1] + sentence_lengths[i-1]

    for mention_list_per_entity in mention_pos:
        adjusted_mentions_per_entity = []
        for mention in mention_list_per_entity:
            men_start, men_end, ent_id, men_sen_id, men_index, men_index_plus_entity_number = mention
            
            # Lấy tổng độ dài của các câu trước câu hiện tại
            offset = cumulative_lengths[men_sen_id]
            
            # Điều chỉnh men_start và men_end
            new_men_start = men_start - offset
            new_men_end = men_end - offset
            
            adjusted_mentions_per_entity.append((new_men_start, new_men_end, ent_id, men_sen_id, men_index, men_index_plus_entity_number))
        adjusted_mention_pos.append(adjusted_mentions_per_entity)
        
    return adjusted_mention_pos
def transform_document_data(relations, mention_pos, hts, sents, pmid):

    document = {
        "title": pmid,
        "sents": sents,
        "labels": [],
        "vertexSet": []
    }

    # Process relations and hts to create "labels"
    for i, (head_ent_id, tail_ent_id) in enumerate(hts):
        relation_type = relations[i]

        # Determine 'dist' (CROSS or NON-CROSS)
        # We need to find the sentence IDs of the head and tail entities.
        # This requires iterating through mention_pos to link ent_id to sent_id.
        head_sent_id = -1
        tail_sent_id = -1

        # Find the sentence ID for the head entity
        for entity_mentions in mention_pos:
            for mention in entity_mentions:
                if mention[2] == head_ent_id:  # mention[2] is ent_id
                    head_sent_id = mention[3]  # mention[3] is men_sen_id
                    break
            if head_sent_id != -1:
                break

        # Find the sentence ID for the tail entity
        for entity_mentions in mention_pos:
            for mention in entity_mentions:
                if mention[2] == tail_ent_id:  # mention[2] is ent_id
                    tail_sent_id = mention[3]  # mention[3] is men_sen_id
                    break
            if tail_sent_id != -1:
                break

        dist = "NON-CROSS" if head_sent_id == tail_sent_id else "CROSS"

        document["labels"].append({
            "h": head_ent_id,
            "t": tail_ent_id,
            "r": relation_type,
            "dist": dist
        })

    # Process mention_pos to create "vertexSet"
    # Group mentions by ent_id
    entities_by_id = {}
    for entity_mentions in mention_pos:
        for mention_tuple in entity_mentions:
            # (men_start, men_end, ent_id, men_sen_id, men_index, men_index+entity_number)
            men_start, men_end, ent_id, men_sen_id, _, _ = mention_tuple
            
            if ent_id not in entities_by_id:
                entities_by_id[ent_id] = []
            
            # Construct the 'name' from the sentence words
            name_words = sents[men_sen_id][men_start:men_end]
            
            entities_by_id[ent_id].append({
                "pos": [men_start, men_end],
                "type": ent_id,  # Assuming 'type' can be the ent_id itself, or you might have a mapping
                "sent_id": men_sen_id,
                "name": name_words
            })
    sorted_entity_ids = sorted(entities_by_id.keys())
    for ent_id in sorted_entity_ids:
        document["vertexSet"].append(entities_by_id[ent_id])

    return document

def read_biored(file_in):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    max_entity = 0
    maxlen = 0
    entity_num = []
    entity_1 = []
    entity_2 = []
    entity_3 = []
    entity_4 = []
    all_documents_data = []
    tmppp = 0
    if file_in == "":
        return None

    with open(file_in, "r") as fh:
        data = json.load(fh)
    """
    'input_ids': input_ids,
    'entity_pos': entity_pos,
    'labels': relations,
    'hts': hts,
    'title': sample['title'],
    'dists': dists,
    """
    
    document = data["documents"]
    for sample in tqdm(document, desc="Example"):
        if len(sample["relations"]) == 0:
            continue
        sent = ''
        sentsss = ''
        sent_map = []
        entity_id = []
        mention_pos = []
        men_ent_list = []
        for i in range(50):
            men_ent_list.append([])
            mention_pos.append([])
        relations = []
        train_triples = {}
        hts = []
        pmid = int(sample["id"])
        for id, text in enumerate(sample["passages"]):
            sent += text["text"]
        sents = [t.split(' ') for t in sent.split('|')]
        sents[-1] = [t for t in sents[-1] if t != '']
        sentss = [t for t in sent.split('|')]
        for t in sentss:
            sentsss = ' '.join([sentsss,t])
        sentsss = sentsss.strip()
        sentssss = sentsss.split()
        total_len = 0
        for i in range(len(sents)):
            length = len(sents[i])
            sent_map.append([total_len, total_len+length])
            total_len += length
        entity_number = 0
        men_ent_list1 = []
        for i in range(50):
            men_ent_list1.append([])
        for id, text in enumerate(sample["passages"]):
            for index, men in enumerate(text["annotations"]):
                men_text = men["text"]
                men_words = men_text.split()
                men_id = int(men["id"])
                men_ent_id = men["infons"]["identifier"]
                men_ent_type = men["infons"]["type"]
                if ',' in men_ent_id:
                    men_ent_ids = men_ent_id.split(",")
                    for id in men_ent_ids:
                        if id not in entity_id:
                            entity_id.append(id)
                            ent_id = entity_id.index(id)
                            men_ent_list1[ent_id].append(men_id)
                        else:
                            ent_id = entity_id.index(id)
                            men_ent_list1[ent_id].append(men_id)
                else:
                    if men_ent_id not in entity_id:
                        entity_id.append(men_ent_id)
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list1[ent_id].append(men_id)
                    else:
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list1[ent_id].append(men_id)
        men_ent_list1 = [t for t in men_ent_list1 if t != []]
        entity_number = len(men_ent_list1)
        docu = ''
        men_text_occur = {}
        search_pos = 0
        men_index = 0
        for id, text in enumerate(sample["passages"]):
            for index, men in enumerate(text["annotations"]):
                men_text = men["text"]
                men_words = men_text.split()
                men_id = int(men["id"])

                men_ent_id = men["infons"]["identifier"]
                men_ent_type = men["infons"]["type"]
                if ',' in men_ent_id:
                    men_ent_ids = men_ent_id.split(",")
                    for id in men_ent_ids:
                        if id not in entity_id:
                            entity_id.append(id)
                            ent_id = entity_id.index(id)
                            men_ent_list[ent_id].append(men_id)
                        else:
                            ent_id = entity_id.index(id)
                            men_ent_list[ent_id].append(men_id)
                else:
                    if men_ent_id not in entity_id:
                        entity_id.append(men_ent_id)
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list[ent_id].append(men_id)
                    else:
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list[ent_id].append(men_id)
                a = True
                while a:
                    men_start = sentssss.index(men_words[0], search_pos)
                    men_ending = men_start + len(men_words) - 1
                    if sentssss[men_ending] == men_words[-1]:
                        men_end = men_start + len(men_words)
                        a = False
                    else:
                        search_pos = men_start + 1 
                for pos in sent_map:
                    if men_start >= pos[0] and men_end < pos[1] and sent_map.index(pos) <= len(sent_map) - 1:
                        men_sen_id = sent_map.index(pos)
                        break
                    elif sent_map.index(pos) == len(sent_map) - 1:
                        print("men_words:",men_words)
                        print("sentssss:", sentsss)
                        print("start,pos:", men_start, men_end)
                        print("sent_map:", sent_map)
                        print("pos:", pos)
                        print("sent_map_len, pos:", len(sent_map), sent_map.index(pos))
                        print("can not find sentence id of mention")
                        return 0
                    else:
                        continue
                if ',' in men_ent_id:
                    for id in men_ent_ids:
                        ent_id = entity_id.index(id)
                        mention_pos[ent_id].append((men_start, men_end, ent_id, men_sen_id, men_index, men_index+entity_number))
                else:
                    ent_id = entity_id.index(men_ent_id)
                    mention_pos[ent_id].append((men_start, men_end, ent_id, men_sen_id, men_index, men_index+entity_number))
                search_pos = men_end
                men_index += 1

        men_ent_list = [t for t in men_ent_list if t != []]
        mention_pos = [x for x in mention_pos if x != []]
        entity_pos = []
        entity_node = []
        mention_node = []
        for i in range(len(mention_pos)):
            entity_node +=[[i,i,i,i,i,i,0]]
            for men in mention_pos[i]:
                entity_pos.append(men)
                mention_node += [list(men) + [1]]
        for id, label in enumerate(sample["relations"]):
            # if len(sample["relations"])==0:
            #     print("no relation instances")
            #     continue
            infons = label["infons"]
            ent_1 = infons["entity1"]
            ent_2 = infons["entity2"]
            relation = [0] * len(bio_rel2id)
            r = bio_rel2id[infons["type"]]
            relation[r] = 1
            ent_1_id = entity_id.index(ent_1)
            ent_2_id = entity_id.index(ent_2)
            if (ent_1_id, ent_2_id) not in train_triples:
                train_triples[(ent_1_id, ent_2_id)] = [{'relation': r}]
            else:
                train_triples[(ent_1_id, ent_2_id)].append({'relation': r})
            hts.append([ent_1_id, ent_2_id])
            relations.append(relation)
            pos_samples += 1
        relations = np.argmax(relations, axis=1)
        relations = [int(r) for r in relations]

        mention_pos = adjust_mention_positions(mention_pos, sents)
        all_documents_data.append(transform_document_data(relations, mention_pos, hts, sents, pmid))
    output_file_name = "2"+file_in
    try:
        with open(output_file_name, 'w', encoding='utf-8') as f:
            # json.dumps chuyển đổi danh sách Python thành chuỗi JSON
            # indent=4 để format JSON cho dễ đọc
            # ensure_ascii=False để hỗ trợ các ký tự non-ASCII (nếu có)
            json.dump(all_documents_data, f, indent=4, ensure_ascii=False)
        print(f"\nĐã lưu tất cả các tài liệu vào file: {output_file_name}")
    except IOError as e:
        print(f"Lỗi khi ghi file {output_file_name}: {e}")
    return
read_biored("Dev.BioC_modified.JSON")
read_biored("Test.BioC_modified.JSON")
read_biored("Train.BioC_modified_end.JSON")

import json

# Định nghĩa bảng ánh xạ
bio_rel2id = {
    'Na': 0, 
    'Association': 1, 
    'Positive_Correlation': 2, 
    'Negative_Correlation': 3, 
    'Bind': 4, 
    'Drug_Interaction': 5, 
    'Cotreatment': 6, 
    'Comparison': 7, 
    'Conversion': 8
}

# Đảo ngược dictionary để dễ dàng tìm kiếm key từ value
id2bio_rel = {v: k for k, v in bio_rel2id.items()}

def process_json_file(input_filepath, output_filepath):
    """
    Tải dữ liệu từ file JSON, thay thế giá trị 'r', và lưu lại vào file mới.

    Args:
        input_filepath (str): Đường dẫn đến file JSON đầu vào.
        output_filepath (str): Đường dẫn đến file JSON đầu ra.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn '{input_filepath}'")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: Không thể giải mã file JSON tại đường dẫn '{input_filepath}'. Đảm bảo file có định dạng JSON hợp lệ.")
        return

    # Duyệt qua dữ liệu và thay thế 'r'
    for item in data:
        if "labels" in item and isinstance(item["labels"], list):
            for label in item["labels"]:
                if "r" in label and label["r"] in id2bio_rel:
                    label["r"] = id2bio_rel[label["r"]]

    # Lưu dữ liệu đã sửa đổi vào file mới
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Dữ liệu đã được xử lý và lưu thành công vào '{output_filepath}'")
    except IOError:
        print(f"Lỗi: Không thể ghi file vào đường dẫn '{output_filepath}'")



# Gọi hàm để xử lý file
input_json_file = '2Dev.BioC_modified.JSON'
output_json_file = 'dev.json'
process_json_file(input_json_file, output_json_file)
input_json_file = '2Test.BioC_modified.JSON'
output_json_file = 'test.json'
process_json_file(input_json_file, output_json_file)
input_json_file = '2Train.BioC_modified_end.JSON'
output_json_file = 'train.json'
process_json_file(input_json_file, output_json_file)
import os

os.remove('2Dev.BioC_modified.JSON')
os.remove('2Train.BioC_modified_end.JSON')
os.remove('2Test.BioC_modified.JSON')