import argparse

from utils import *
import os
import socket
from config import *
from multiprocessing import Pool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body, to_email):
    from_email = "xxx@qq.com"
    password = "xxx"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.qq.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")


class PolicyEntity:
    def __init__(self, score) -> None:
        self.score = score

    def get_score(self):
        return self.score


class CardinalityGuidedEntity(PolicyEntity):
    def __init__(self, score, card_str) -> None:
        super().__init__(score)
        self.card_str = card_str


class PgHelper():
    def __init__(self, queries, output_query_latency_file) -> None:
        self.queries = queries
        self.output_query_latency_file = output_query_latency_file

    def start(self, pool_num):
        pool = Pool(pool_num)
        for fp, q in self.queries:
            # Qihan we don't need to use this
            pool.apply_async(do_run_query, args=(q, fp, [], self.output_query_latency_file, True, None, None))
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()


class LeroHelper():
    def __init__(self, queries, query_num_per_chunk, output_query_latency_file, 
                test_queries, model_prefix, topK) -> None:
        self.queries = queries
        self.query_num_per_chunk = query_num_per_chunk
        self.output_query_latency_file = output_query_latency_file
        self.test_queries = test_queries
        self.model_prefix = model_prefix
        self.topK = topK
        self.lero_server_path = LERO_SERVER_PATH
        self.lero_card_file_path = os.path.join(LERO_SERVER_PATH, LERO_DUMP_CARD_FILE)

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def start(self, pool_num):
        lero_chunks = list(self.chunks(self.queries, self.query_num_per_chunk))

        run_args = self.get_run_args()
        use_second_db =False
        for c_idx, chunk in enumerate(lero_chunks):
            #qihan;add this to show the iteration number
            print("iteration = ", c_idx+1)

            if c_idx % 5 == 0 and c_idx != 0:
                use_second_db = not use_second_db

            pool = Pool(pool_num)
            for fp, q in chunk:
                self.run_pairwise(q, fp, run_args, self.output_query_latency_file, self.output_query_latency_file + "_exploratory", pool,use_second_db)
            print('Waiting for all subprocesses done...')
            pool.close()
            pool.join()

            model_name = self.model_prefix + "_" + str(c_idx)
            self.retrain(model_name)
            self.test_benchmark(self.output_query_latency_file + "_" + model_name,use_second_db)

    def retrain(self, model_name):
        training_data_file = self.output_query_latency_file + ".training"
        create_training_file(training_data_file, self.output_query_latency_file, self.output_query_latency_file + "_exploratory")
        print("retrain Lero model:", model_name, "with file", training_data_file)
        
        cmd_str = "cd " + self.lero_server_path + " && python3.8 train.py" \
                                                + " --training_data " + os.path.abspath(training_data_file) \
                                                + " --model_name " + model_name \
                                                + " --training_type 1"
        print("run cmd:", cmd_str)
        os.system(cmd_str)

        self.load_model(model_name)
        return model_name

    def load_model(self, model_name):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        json_str = json.dumps({"msg_type":"load", "model_path": os.path.abspath(LERO_SERVER_PATH + model_name)})
        print("load_model", json_str)

        s.sendall(bytes(json_str + "*LERO_END*", "utf-8"))
        reply_json = s.recv(1024)
        s.close()
        print(reply_json)
        os.system("sync")

    def test_benchmark(self, output_file,use_second_db):
        run_args = self.get_run_args()
        for (fp, q) in self.test_queries:
       
            do_run_query(q, fp, run_args, output_file, True, None, None,True,use_second_db)

    def get_run_args(self):
        run_args = []
        run_args.append("SET enable_lero TO True")
        return run_args

    def get_card_test_args(self, card_file_name):
        run_args = []
        run_args.append("SET lero_joinest_fname TO '" + card_file_name + "'")
        return run_args
    # TODO
    def run_pairwise(self, q, fp, run_args, output_query_latency_file, exploratory_query_latency_file, pool,use_second_db):
        explain_query(q, run_args)
        policy_entities = []
        with open(self.lero_card_file_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split(";") for line in lines]
            for line in lines:
                policy_entities.append(CardinalityGuidedEntity(float(line[1]), line[0]))

        policy_entities = sorted(policy_entities, key=lambda x: x.get_score())
        policy_entities = policy_entities[:self.topK]

        i = 0
        for entity in policy_entities:
            if isinstance(entity, CardinalityGuidedEntity):
                card_str = "\n".join(entity.card_str.strip().split(" "))
                # ensure that the cardinality file will not be changed during planning
                card_file_name = "lero_" + fp + "_" + str(i) + ".txt"
                card_file_path = os.path.join(PG_DB_PATH, card_file_name)
                with open(card_file_path, "w") as card_file:
                    card_file.write(card_str)

                output_file = output_query_latency_file if i == 0 else exploratory_query_latency_file
                
                pool.apply_async(do_run_query, args=(q, fp, self.get_card_test_args(card_file_name), output_file, True, None, None, True, use_second_db))
                i += 1

    def predict(self, plan):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        s.sendall(bytes(json.dumps({"msg_type":"predict", "Plan":plan}) + "*LERO_END*", "utf-8"))
        reply_json = json.loads(s.recv(1024))
        assert reply_json['msg_type'] == 'succ'
        s.close()
        print(reply_json)
        os.system("sync")
        return reply_json['latency']

def read_queries_from_folder(folder_path):
    queries = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.sql'):
            with open(file_path, 'r') as f:
                query = f.read().strip()
                queries.append((file_name, query))
    return queries

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--query_folder",
                        metavar="FOLDER",
                        help="Load the training queries from folder")
    parser.add_argument("--test_query_folder",
                        metavar="FOLDER",
                        help="Load the test queries from folder")
    parser.add_argument("--algo", type=str)
    parser.add_argument("--query_num_per_chunk", type=int)
    # qihan: add iteration
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--output_query_latency_file", metavar="PATH")
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--pool_num", type=int)
    parser.add_argument("--topK", type=int)
    args = parser.parse_args()

    query_folder = args.query_folder
    print("Load queries from folder:", query_folder)
    queries = read_queries_from_folder(query_folder)
    print("Read", len(queries), "training queries.")

    output_query_latency_file = args.output_query_latency_file
    print("output_query_latency_file:", output_query_latency_file)

    pool_num = 10
    if args.pool_num:
        pool_num = args.pool_num
    print("pool_num:", pool_num)

    ALGO_LIST = ["lero", "pg"]
    algo = "lero"
    if args.algo:
        assert args.algo.lower() in ALGO_LIST
        algo = args.algo.lower()
    print("algo:", algo)

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    if algo == "pg":
        helper = PgHelper(queries, output_query_latency_file)
        helper.start(pool_num)
    else:
        test_queries = []
        if args.test_query_folder is not None:
            test_queries = read_queries_from_folder(args.test_query_folder)
        print("Read", len(test_queries), "test queries.")

        #query_num_per_chunk = args.query_num_per_chunk
        #qihan: modify this to use all train query in one iteration
        query_num_per_chunk = len(queries)
        print("query_num_per_chunk:", query_num_per_chunk)

        model_prefix = None
        if args.model_prefix:
            model_prefix = args.model_prefix
        print("model_prefix:", model_prefix)

        topK = 5
        if args.topK is not None:
            topK = args.topK
        print("topK", topK)

        iteration_num = 100
        if args.iteration_num is not None:
            iteration_num = args.iteration_num

        # expand queries to iteration_num times
        queries = queries * iteration_num
        
        helper = LeroHelper(queries, query_num_per_chunk, output_query_latency_file, test_queries, model_prefix, topK)
        helper.start(pool_num)

        send_email("Lero Experiment", "The experiment of tpch switching db finished!", "xxx@qq.com")
