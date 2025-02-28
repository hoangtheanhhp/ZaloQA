import tensorflow as tf
from os.path import join, exists
from .preprocess import ZaloDatasetProcessor
from .modeling import BertClassifierModel
from bert import tokenization
import kashgari
from kashgari.tasks.classification import BiGRU_Model
from kashgari.embeddings import BERTEmbedding, TransformerEmbedding
from kashgari.tokenizer import BertTokenizer
from kashgari.tasks.classification import CNNLSTMModel

import logging
logging.basicConfig(level='DEBUG')

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", None,
                    "Training or Predicting?")
flags.DEFINE_string("dataset_path", None,
                    "The path to the dataset")
flags.DEFINE_string("bert_model_path", None,
                    "Link to BERT cased model")
flags.DEFINE_string("model_path", None,
                    "Default path to store the trained model")

flags.DEFINE_string("train_filename", "train.json",
                    "The name of the training file (stored in the dataset folder)")
flags.DEFINE_string("train_augmented_filename", None,
                    "The name of the additional training file with augmented data (stored in the dataset folder)")
flags.DEFINE_string("dev_filename", None,
                    "The name of the developemt file (stored in the dataset folder)")
flags.DEFINE_string("test_filename", "test.json",
                    "The name of the testing file (stored in the dataset folder)")
flags.DEFINE_string("test_predict_outputmode", "zalo",
                    "The mode in which the predict file should be (Zalo-defined 'zalo' or full information 'full')")

flags.DEFINE_integer("max_sequence_len", 256,
                     "The maximum input sequence length for embeddings")
flags.DEFINE_bool("do_lowercase", False,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")
flags.DEFINE_float("model_learning_rate", 1e-5,
                   "The default model learning rate")
flags.DEFINE_integer("model_batch_size", 16,
                     "Training input batch size")
flags.DEFINE_integer("train_epochs", 3,
                     "Number of loops to train the whole dataset")
flags.DEFINE_float("train_dropout_rate", 0.1,
                   "Default dropout rate")
flags.DEFINE_float("bert_warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup")
flags.DEFINE_bool("use_pooled_output", True,
                  "Use pooled output from pretrained BERT. False for using meaned output")

flags.DEFINE_string("loss_type", "cross_entropy",
                    "The default loss function to use during training (Can be *cross_entropy* or *focal_loss*")
flags.DEFINE_float("loss_label_smooth", 0,
                   "Float in [0, 1] to perform label smoothing when calculate loss. "
                   "When 0, no smoothing occurs. When positive, the binary"
                   "ground truth labels `y_true` are squeezed toward 0.5, with larger values"
                   "of `label_smoothing` leading to label values closer to 0.5.")

flags.DEFINE_integer("save_checkpoint_steps", 500,
                     "The number of steps between each checkpoint save")
flags.DEFINE_integer("save_summary_steps", 100,
                     "The number of steps between each summary write")
flags.DEFINE_integer("keep_checkpoint_max", 1,
                     "The maximum number of checkpoints to keep")

flags.DEFINE_string("encoding", "utf-8",
                    "Encoding used in the dataset")
flags.DEFINE_string("zalo_predict_csv_file", "./zalo.csv",
                    "Destination for the Zalo submission predict file")
flags.DEFINE_string("eval_predict_csv_file", None,
                    "Destination for the development set predict file (None if no output is required)")
flags.DEFINE_float("dev_size", 0.2,
                   "The size of the development set taken from the training set"
                   "If dev_filename exists, this is ignored")


def main(_):
    print("[Main] Starting....")

    # Tokenizer initialzation
    tokenizer = BertTokenizer.load_from_vacob_file(vocab_path)
    embed = TransformerEmbedding(vocab_path, config_path, checkpoint_path,
                                 bert_type='bert',
                                 task=kashgari.CLASSIFICATION,
                                 sequence_length=FLAGS.max_sequence_len)
    processor = ZaloDatasetProcessor()
    processor.load_from_path("")
    model = CNNLSTMModel(embed)
    model.evaluate()

    # Training/Testing
    # if FLAGS.mode.lower() == 'train':
    #     print('[Main] Begin training')
    #     eval_result = model.train_and_eval()
    #     print('[Main] Training complete.')
    #     print('[Main] Evaluation complete')
    #     print("Accuracy: {}%".format(eval_result['accuracy'] * 100))
    #     print("Loss: {}".format(eval_result['loss']))
    #     print("F1 Score: {}".format(eval_result['f1_score'] * 100))
    #     print("Recall: {}%".format(eval_result['recall'] * 100))
    #     print("Precision: {}%".format(eval_result['precision'] * 100))
    #     if FLAGS.eval_predict_csv_file is not None:
    #         print('[Main] Development set predict and output to file')
    #         _ = model.predict_from_eval_file(test_file=dev_file, output_file=FLAGS.eval_predict_csv_file,
    #                                          file_output_mode='full')
    # elif FLAGS.mode.lower() == 'eval':
    #     eval_result = model.eval()
    #     print('[Main] Evaluation complete')
    #     print("Accuracy: {}%".format(eval_result['accuracy'] * 100))
    #     print("Loss: {}".format(eval_result['loss']))
    #     print("F1 Score: {}".format(eval_result['f1_score'] * 100))
    #     print("Recall: {}%".format(eval_result['recall'] * 100))
    #     print("Precision: {}%".format(eval_result['precision'] * 100))
    #     if FLAGS.eval_predict_csv_file is not None:
    #         print('[Main] Development set predict and output to file')
    #         _ = model.predict_from_eval_file(test_file=dev_file, output_file=FLAGS.eval_predict_csv_file,
    #                                          file_output_mode='full')
    # elif FLAGS.mode.lower() == 'predict_test':
    #     print("[Main] Begin Predict based on Test file")
    #     results = model.predict_from_eval_file(test_file=test_file, output_file=FLAGS.zalo_predict_csv_file,
    #                                            file_output_mode=FLAGS.test_predict_outputmode)
    #     print(results)
    # elif FLAGS.mode.lower() == 'predict_manual':
    #     while True:
    #         question = input("Please enter question here (or empty to exit): ")
    #         if question == "":
    #             break
    #         paragragh = input("Please enter potential answer here here (or empty to exit): ")
    #         if paragragh == "":
    #             break
    #         result = model.predict([(question, paragragh)])[0]
    #         print('Prediction: {} with confidence of {}%'
    #               .format(result['prediction'], result['probabilities'] * 100))
    #
    # print('[Main] Finished')


if __name__ == "__main__":
    """ Sanity flags check """
    assert FLAGS.mode.lower() in ['train', 'eval', 'predict_test', 'predict_manual'], \
        "[FlagsCheck] Mode can only be 'train', 'eval', 'predict_test' or 'predict_manual'"
    assert exists(FLAGS.dataset_path), "[FlagsCheck] Dataset path doesn't exist"
    assert exists(FLAGS.bert_model_path), "[FlagsCheck] BERT pretrained model path doesn't exist"
    assert FLAGS.test_predict_outputmode.lower() in ['full', 'zalo'], "[FlagsCheck] Test file output mode " \
                                                                      "can only be 'full' or 'zalo'"
    assert FLAGS.model_path is not None, "[FlagsCheck] BERT finetuned model location must be set"
    assert FLAGS.loss_type.lower() in ['cross_entropy', 'focal_loss', 'kld', 'squared_hinge', 'hinge'],\
        "[FlagsCheck] Incorrect loss function used"
    tf.compat.v1.app.run()
