error_file_count = 0
# citec
data_src_path = "/media/remote/tmandal/document-clustering-with-deep-neural-networks/data/QS-OCR-Large/"
# pop_os!
# data_src_path = "/home/tma/project/document-clustering-with-deep-neural-networks/data/QS-OCR-Large/"
# citec
prj_src_path = "/media/remote/tmandal/document-clustering-with-deep-neural-networks/"
# pop_os!
# prj_src_path = "/home/tma/project/document-clustering-with-deep-neural-networks/"
test_label_file_name = "text_test.txt"
train_label_file_name = "text_train.txt"
val_label_file_name = "text_val.txt"
label_name = {"0": "Letter", "1": "Form", "2": "Email", "3": "", "4": "Advertisement", "5": "Scientific report",
              "6": "Scientific publication", "7": "Specification", "8": "File folder", "9": "News article",
              "10": "Budget", "11": "Invoice", "12": "Presentation", "13": "Questionnaire", "14": "Resume",
              "15": "Memo"}
translation = {"0": 0, "1": 1, "2": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9, "11": 10, "12": 11,
               "13": 12, "14": 13, "15": 14}
translation_rev = ["0", "1", "2", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
