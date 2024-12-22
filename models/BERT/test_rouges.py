from others.cal_rouge import cal_rouge_path
result_path="logs/bert_abs_final"
step=3500
gold_path = result_path + '.%d.gold' % step
can_path = result_path + '.%d.candidate' % step
if __name__=="__main__":
    cal_rouge_path("./logs",can_path, gold_path)