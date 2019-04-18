# coding=utf-8
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    # Data input settings
    parser.add_argument('--TRAIN_GPU_ID', type=int, default=0)
    parser.add_argument('--TEST_GPU_ID', type=int, default=0)
    parser.add_argument('--SEED', type=int, default=-1)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=1)
    parser.add_argument('--NUM_OUTPUT_UNITS', type=int, default=3000)
    parser.add_argument('--MAX_WORDS_IN_QUESTION', type=int, default=33)
    parser.add_argument('--MAX_WORDS_IN_RELATION', type=int, default=6)
    parser.add_argument('--MAX_ITERATIONS', type=int, default=100000)
    parser.add_argument('--PRINT_INTERVAL', type=int, default=100)
    parser.add_argument('--CHECKPOINT_INTERVAL', type=int, default=500)
    parser.add_argument('--TESTDEV_INTERVAL', type=int, default=45000)
    parser.add_argument('--RESUME', type=bool, default=False)
    parser.add_argument('--RESUME_PATH', type=str, default='./data/***.pth')
    parser.add_argument('--VAL_INTERVAL', type=int, default=5000)
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=2048)
    parser.add_argument('--INIT_LEARNING_RATE', type=float, default=0.0001)
    parser.add_argument('--DECAY_STEPS', type=int, default=20000)
    parser.add_argument('--DECAY_RATE', type=float, default=0.5)
    parser.add_argument('--MFB_FACTOR_NUM', type=int, default=5)
    parser.add_argument('--MFB_OUT_DIM', type=int, default=1000)
    parser.add_argument('--LSTM_UNIT_NUM', type=int, default=1024)
    parser.add_argument('--LSTM_DROPOUT_RATIO', type=float, default=0.3)
    parser.add_argument('--MFB_DROPOUT_RATIO', type=float, default=0.1)
    parser.add_argument('--TRAIN_DATA_SPLITS', type=str, default='train')
    parser.add_argument('--QUESTION_VOCAB_SPACE', type=str, default='train')
    parser.add_argument('--RELATION_VOCAB_SPACE', type=str, default='train')
    parser.add_argument('--save_path', type=str, default='id1/results4')
    parser.add_argument('--pth_path', type=str, default='id1/train_pth4')
    parser.add_argument('--fb_file', type=str, default=
                        "/home/chenshuo/.pyenvs/KBQApy35/BuboQA/data/SimpleQuestions_v2/freebase-subsets/processed-FB2M.txt")
    parser.add_argument('--el_file', type=str, default= "/home/chenshuo/.pyenvs/KBQApy35/BuboQA/entity_linking/char_revising_results/lstm")
    parser.add_argument('--dataset_file', type=str, default="/home/chenshuo/.pyenvs/KBQApy35/BuboQA/data/processed_simplequestions_dataset")
    parser.add_argument('--raw_fb_file', type=str, default= "/home/chenshuo/.pyenvs/KBQApy35/BuboQA/data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt")

    args = parser.parse_args(args=[])
    return args
