import copy
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np
from rouge import Rouge
from tqdm import tqdm

rouge = Rouge()
org_data_path = 'data/csds/'
new_data_path = 'json_spe_contrast/csds/'


def convert(data, is_train=True):
    new_data = []
    new_data_user = []
    new_data_agent = []
    new_data_final = []
    num = 0
    for i in range(3):
        for sample in tqdm(data):
            num = num + 1
            qrole = sample['QRole']
            new_sample = {'type': i, 'session': [], 'summary': []}
            consecutive_id_map = {}
            previous_speaker = ""
            tmp_utt = []
            tmp_utt_id = []
            turn_num = 0
            for j, turn in enumerate(sample['Dialogue']):
                utterance = turn['utterance'].replace(' ', '')
                utterance = re.sub(r'(http|https)?://[a-zA-Z0-9(\[数字\])(\[链接\]).?/&=:\-#\+]*', '[网址]', utterance)
                if turn["speaker"] != previous_speaker:
                    if tmp_utt != []:
                        utter = {}
                        if turn["speaker"] == "Q":
                            utter['type'] = 'agent'
                            utter['content'] = "客服：" + ''.join(tmp_utt)
                        else:
                            utter['type'] = 'user'
                            utter['content'] = qrole + "：" + ''.join(tmp_utt)
                        new_sample['session'].append(utter)
                        for id in tmp_utt_id:
                            consecutive_id_map[id] = turn_num
                        turn_num = turn_num + 1
                    tmp_utt = []
                    tmp_utt_id = []
                    tmp_utt.append(utterance)
                    tmp_utt_id.append(j)
                else:
                    tmp_utt.append(utterance)
                    tmp_utt_id.append(j)
                previous_speaker = turn["speaker"]
            if tmp_utt != []:
                utter = {}
                if previous_speaker == 'Q':
                    utter['type'] = 'user'
                    utter['content'] = qrole + "：" + ''.join(tmp_utt)
                else:
                    utter['type'] = 'agent'
                    utter['content'] = '客服：' + ''.join(tmp_utt)
                new_sample['session'].append(utter)
                for id in tmp_utt_id:
                    consecutive_id_map[id] = turn_num
                turn_num = turn_num + 1

            if i == 0:
                user_sum=sample['UserSumm']
                new_sample['summary'] = user_sum
            elif i == 1:
                agent_sum=sample['AgentSumm']
                new_sample['summary'] = agent_sum
            elif i == 2:
                final_sum = sample['FinalSumm']
                new_sample['summary'] = final_sum
            assert len(sample['UserSumm']) == len(sample['AgentSumm'])
            new_sample['user_session'] = [utt for utt in new_sample['session'] if utt['type'] == "user"]
            new_sample['agent_session'] = [utt for utt in new_sample['session'] if utt['type'] == 'agent']
            new_data.append(copy.deepcopy(new_sample))
            if i == 0:
                new_data_user.append(copy.deepcopy(new_sample))
            elif i == 1:
                new_data_agent.append(copy.deepcopy(new_sample))
            elif i == 2:
                new_data_final.append(copy.deepcopy(new_sample))
    #new_data = negative_sample_construction(new_data, 3)
    #new_data = negative_salient_sample_construction(new_data, 3)
    #new_data = negative_shuffle_sample_construction(new_data, 3)
    if is_train:
        new_data = negative_consecutive_sample_construction(new_data, 3)
    return new_data, new_data_user, new_data_agent, new_data_final



def negative_consecutive_sample_construction(new_data, num):
    for i in range(len(new_data)):
        session = new_data[i]['session']
        expand_corrupt_user_sessions = []
        expand_corrupt_agent_sessions = []
        vis_user = defaultdict(int)
        cnt=len(new_data)//3
        vis_user[i] = 1

        vis_agent = defaultdict(int)
        vis_agent[i] = 1
        if (i +cnt) < len(new_data):
            vis_user[i + cnt] = 1
            vis_agent[i+cnt]=1
        if (i + 2 * cnt) < len(new_data):
            vis_user[i + 2 * cnt] = 1
            vis_agent[i + 2 * cnt] = 1
        if (i -cnt) > 0:
            vis_user[i - cnt] = 1
            vis_agent[i - cnt] = 1
        if (i - 2 * cnt) > 0:
            vis_user[i - 2 * cnt] = 1
            vis_agent[i - 2 * cnt] = 1

        for k in range(num):
            corrupt_user_session = []
            corrupt_agent_session = []
            random_user_id=random.randint(0,len(new_data)-1)
            while(vis_user[random_user_id]):
                random_user_id = random.randint(0, len(new_data) - 1)

            random_user_new_data=new_data[random_user_id]
            random_data_user_session =random_user_new_data['user_session']
            random_user_num = len(random_data_user_session)

            vis_user[random_user_id]=1
            if (random_user_id + cnt) < len(new_data):
                vis_user[random_user_id + cnt] = 1

            if (random_user_id + 2 * cnt) < len(new_data):
                vis_user[random_user_id + 2 * cnt] = 1

            if (random_user_id - cnt) > 0:
                vis_user[random_user_id - cnt] = 1

            if (random_user_id - 2 * cnt) > 0:
                vis_user[random_user_id - 2 * cnt] = 1

            user_count = 0
            for utt in session:
                role = utt['type']
                if role == "user":
                    if user_count < random_user_num:
                        corrupt_user_session.append(random_data_user_session[user_count])
                        user_count = user_count + 1
                    else:
                        random_user_id = random.randint(0, len(new_data) - 1)
                        while (vis_user[random_user_id] ):
                            random_user_id = random.randint(0, len(new_data) - 1)
                        random_user_new_data = new_data[random_user_id]
                        random_data_user_session = random_user_new_data['user_session']
                        random_user_num = len(random_data_user_session)
                        vis_user[random_user_id] = 1
                        user_count = 0
                        corrupt_user_session.append(random_data_user_session[user_count])
                        user_count = user_count + 1
                else:
                    corrupt_user_session.append(utt)

            expand_corrupt_user_sessions.append(corrupt_user_session)
            random_agent_id = random.randint(0, len(new_data) - 1)

            while (vis_agent[random_agent_id]):
                random_agent_id = random.randint(0, len(new_data) - 1)
            random_agent_new_data = new_data[random_agent_id]
            random_data_agent_session = random_agent_new_data['agent_session']
            random_agent_num = len(random_data_agent_session)
            vis_agent[random_agent_id] = 1
            if (random_agent_id + cnt) < len(new_data):
                vis_agent[random_agent_id + cnt] = 1

            if (random_agent_id + 2 * cnt) < len(new_data):
                vis_agent[random_agent_id + 2 * cnt] = 1

            if (random_agent_id - cnt) > 0:
                vis_agent[random_agent_id - cnt] = 1

            if (random_agent_id - 2 * cnt) > 0:
                vis_agent[random_agent_id - 2 * cnt] = 1
            agent_count = 0
            for utt in session:
                role = utt['type']
                if role == "agent":
                    if agent_count < random_agent_num:
                        corrupt_agent_session.append(random_data_agent_session[agent_count])
                        agent_count = agent_count + 1
                    else:
                        random_agent_id = random.randint(0, len(new_data) - 1)
                        while (vis_agent[random_agent_id]):
                            random_agent_id = random.randint(0, len(new_data) - 1)
                        random_agent_new_data = new_data[random_agent_id]
                        random_data_agent_session = random_agent_new_data['agent_session']
                        random_agent_num = len(random_data_agent_session)
                        vis_agent[random_agent_id] = 1
                        agent_count = 0
                        corrupt_agent_session.append(random_data_agent_session[agent_count])
                        agent_count = agent_count + 1
                else:
                    corrupt_agent_session.append(utt)
            expand_corrupt_agent_sessions.append(corrupt_agent_session)
        new_data[i]['contrast_user_sessions'] = copy.deepcopy(expand_corrupt_user_sessions)
        new_data[i]["contrast_agent_sessions"] = copy.deepcopy(expand_corrupt_agent_sessions)
    return new_data





if __name__ == '__main__':
    # for mode in ['final', 'user', 'agent']:
    for name in ['train', 'val', 'test']:
        with open(org_data_path + name + '.json', 'r', encoding="utf-8") as f:
            data = json.load(f)
        if name=="train":
            new_data, new_data_user, new_data_agent, new_data_final = convert(data, is_train=True)
        else:
            new_data, new_data_user, new_data_agent, new_data_final = convert(data, is_train=False)
        with open(new_data_path + 'all' + '/' + 'csds.' + name + '.json', 'w', encoding="utf-8") as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        with open(new_data_path + 'user' + '/' + 'csds.' + name + '.json', 'w', encoding="utf-8") as f:
            json.dump(new_data_user, f, indent=4, ensure_ascii=False)
        with open(new_data_path + 'agent' + '/' + 'csds.' + name + '.json', 'w', encoding="utf-8") as f:
            json.dump(new_data_agent, f, indent=4, ensure_ascii=False)
        with open(new_data_path + 'final' + '/' + 'csds.' + name + '.json', 'w', encoding="utf-8") as f:
            json.dump(new_data_final, f, indent=4, ensure_ascii=False)
