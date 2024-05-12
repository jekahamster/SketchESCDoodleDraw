import argparse
import time


def _str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_opt():

    parser = argparse.ArgumentParser()
    t=time.localtime()

    # Overall settings

    parser.add_argument(
        '--home', '-home',
        type=str,
        default='./')

    parser.add_argument(
        '--ckpt_folder', '-ckpt',
        type=str,
        default='checkpoint/'+str(t.tm_mon)+"-"+str(t.tm_mday)+"-"+str(t.tm_hour)+"-"+str(t.tm_min)+"-"+str(t.tm_sec)+"/")

    parser.add_argument(
        '--log_folder', '-log',
        type=str,
        default='log/'+str(t.tm_mon)+"-"+str(t.tm_mday)+"-"+str(t.tm_hour)+"-"+str(t.tm_min)+"-"+str(t.tm_sec)+"/")

    parser.add_argument(
        '--dataset_path', '-dataset_path',
        type=str,
        default='./data',
        help='Path to the dataset folder'    
    )

    parser.add_argument(
        '--pretrain_path', '-pretrain_path',
        type=str,
        default="./pretrained_model",
        help='Path to the pre-trained model for fine-tuning'    
    )

    parser.add_argument(
        '--lr', '-lr',
        type=float,
        default=3e-4,
        help='Learning rate for Adam optimizer'    
    )

    parser.add_argument(
        '--weight_decay', '-weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for Adam optimizer'
    )

    parser.add_argument(
        '--bs', '-bs',
        type=int,
        default=128,
        help='Batch size for training'
    )

    parser.add_argument(
        '--max_stroke', '-max_stroke',
        type=int,
        default=43,
        help='Maximum number of strokes in a sketch'    
    )

    parser.add_argument(
        '--mask', '-mask',
        type=_str_to_bool, nargs='?', const=True,
        default=False,
        help='Whether to mask the input strokes'    
    )

    parser.add_argument(
        '--shape_emb', '-shape_emb',
        type=str,
        default='sum',
        help='Shape embedding type: sum or ???'
    )

    parser.add_argument(
        '--shape_extractor', '-shape_extractor',
        type=str,
        default='lstm',
        help='Shape extractor type: lstm or ???'
    )

    parser.add_argument(
        '--shape_extractor_layer', '-shape_extractor_layer',
        type=int,
        default=2,
        help='Number of layers for shape extractor'    
    )

    parser.add_argument(
        '--embedding_dropout', '-embedding_dropout',
        type=float,
        default=0,
        help='Dropout rate for embedding layers'
    )

    parser.add_argument(
        '--attention_dropout', '-attention_dropout',
        type=float,
        default=0,
        help='Dropout rate for attention layers')

    parser.add_argument(
        '--gpu', '-gpu',
        type=int,
        default=0,
        help='GPU index for training')

    args = parser.parse_args()

    return args


def parse_inference_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sample_path', '-sample_path',
        type=str,
        required=True,
        help='Path to the sample sketch file in a .json format'
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        help="Path to the dataset folder"
    )

    args = parser.parse_args()

    return args