import json
import json
with open("./data/csds/test.json", 'r', encoding="utf-8") as f:
    data = json.load(f)
def save_new_file(hyp_path,ref_path,lists,save_path):
    preds=[]
    refs=[]
    print(hyp_path)
    with open(hyp_path,"r",encoding="utf-8") as f:
        hyp_lines=f.readlines()

    for line in hyp_lines:
        preds.append(line.strip().replace(" ",""))
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_lines = f.readlines()
    for line in ref_lines:
        refs.append(line.strip().replace(" ", ""))
    print(len(preds))
    print(len(refs))
    new_hyps=[preds[l] for l in lists]
    new_refs=[refs[l] for l in lists]
    new_hyp_path=save_path+"/"+"hyper.txt"
    new_ref_path=save_path+"/"+"ref.txt"
    with open(new_hyp_path,"w",encoding="utf-8") as f:
        for hyp in new_hyps:
            f.write(hyp+"\n")
    with open(new_ref_path,"w",encoding="utf-8") as f:
        for ref in new_refs:
            f.write(ref+"\n")

incomplete_instances=[]
complete_instances=[]
id=0
for sample in data:
    user_utterances=[]
    agent_utterances=[]
    for j, turn in enumerate(sample['Dialogue']):
        if turn["speaker"] == "Q":
            user_utterances.append(j)
        else:
            agent_utterances.append(j)
    qa=sample["QA"]
    flag=0

    print(agent_utterances)
    for pair in qa:
        q_utterances=pair["QueSummUttIDs"]
        a_utterances=pair["AnsSummLongUttIDs"]

        for utt in a_utterances:
            if utt in user_utterances:
                flag=1
    if flag==1:
        incomplete_instances.append(id)
    else:
        complete_instances.append(id)
    id=id+1
print(len(incomplete_instances))
print(len(complete_instances))
save_new_file("./eval_summ/bart_spe_contrast_csds/agent_test_results/hyper.txt",\
                "./eval_summ/bart_spe_contrast_csds/agent_test_results/ref.txt",incomplete_instances,\
                "./eval_summ/bart_spe_contrast_csds/agent_test_results/incomplete")
save_new_file("./eval_summ/bart_spe_contrast_csds/agent_test_results/hyper.txt", \
                "./eval_summ/bart_spe_contrast_csds/agent_test_results/ref.txt", complete_instances, \
                "./eval_summ/bart_spe_contrast_csds/agent_test_results/complete")
save_new_file("./eval_summ/bart_spe_csds/agent_test_results/hyper.txt", \
                "./eval_summ/bart_spe_csds/agent_test_results/ref.txt", incomplete_instances, \
                "./eval_summ/bart_spe_csds/agent_test_results/incomplete")
save_new_file("./eval_summ/bart_spe_csds/agent_test_results/hyper.txt", \
                "./eval_summ/bart_spe_csds/agent_test_results/ref.txt", complete_instances, \
                "./eval_summ/bart_spe_csds/agent_test_results/complete")
save_new_file("./eval_summ/bart_glc_csds/agent_test_results/hyper.txt", \
                "./eval_summ/bart_glc_csds/agent_test_results/ref.txt", incomplete_instances, \
                "./eval_summ/bart_glc_csds/agent_test_results/incomplete")
save_new_file("./eval_summ/bart_glc_csds/agent_test_results/hyper.txt", \
                "./eval_summ/bart_glc_csds/agent_test_results/ref.txt", complete_instances, \
                "./eval_summ/bart_glc_csds/agent_test_results/complete")





