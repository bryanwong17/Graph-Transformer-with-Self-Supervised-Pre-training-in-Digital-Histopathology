import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=2, help='classification classes')
        parser.add_argument('--data_path', type=str, default="D:/MedicalAI/Graph VIT based SSL/build_graphs/simsiam/graphs_tcga_resnet50_not_pretrained_new_3000/simsiam_files", help='path to dataset where images store')
        parser.add_argument('--train_set', type=str, default="slide_label/TCGA_3000/train_set.txt", help='train')
        parser.add_argument('--val_set', type=str, default="slide_label/TCGA_3000/val_set.txt", help='validation')
        parser.add_argument('--model_path', type=str, default="../graph_transformer/results_without_graph/simsiam/tcga_resnet50_not_pretrained_new_3000_BatchSizeSimclr16_4/saved_models", help='path to trained model')
        parser.add_argument('--log_path', type=str, default="../graph_transformer/results_without_graph/simsiam/tcga_resnet50_not_pretrained_new_3000_BatchSizeSimclr16_4/runs", help='path to log files')
        parser.add_argument('--task_name', type=str, default="GraphTransformer", help='task name for naming saved model files and log files')
        parser.add_argument('--train', action='store_false', default=True, help='train only')
        parser.add_argument('--test', action='store_true', default=False, help='test only')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--log_interval_local', type=int, default=10, help='classification classes')
        parser.add_argument('--resume', type=str, default="results_without_graph/simsiam/tcga_resnet50_not_pretrained_new_3000_BatchSizeSimclr16_4/saved_models/GraphTransformer.pth",help='path for model')
        parser.add_argument('--resume', type=str,help='path for model')
        parser.add_argument('--graphcam', action='store_true', default=False, help='GraphCAM')
        parser.add_argument('--figure_path', type=str, default="results_without_graph/simsiam/tcga_resnet50_not_pretrained_new_3000_BatchSizeSimclr16_4/curve", help='GraphCAM')

        # the parsess
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr

        args.num_epochs = 120
        args.lr = 1e-3           

        if args.test:
            args.num_epochs = 1
        return args
