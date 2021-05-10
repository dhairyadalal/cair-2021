import pytrec_eval

def read_qrel_from_file(file_path: str) -> dict:
    """ Method return json representation of qrel file """
    
    qrel_json = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            line_split = line.split("\t")
            query_id = line_split[0]
            doc = line_split[2]
            rel = int(line_split[3].strip())

            if query_id in qrel_json:
                qrel_json[query_id][doc]=rel
            else:
                qrel_json[query_id]={doc:rel}
                
    return qrel_json


def evaluate_run(run: dict, qrel: dict,  metrics: set = {'map', 'ndcg'}) -> dict:
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    return evaluator.evaluate(qrel)


