"""
Train a speaker diarization model
"""
import os
import torch
import numpy as np
import uisrnn
import random

from datetime import datetime

MODEL_NAME_STUB = 'model.UISRNN'

def get_device(use_cuda=True):
    cuda_available = torch.cuda.is_available()
    use_cuda = use_cuda and cuda_available

    # Prompt user to use CUDA if available
    if cuda_available and not use_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Set device
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    print('Device: {}'.format(device))
    if use_cuda:
        print('Using CUDA {}'.format(torch.cuda.current_device()))
    return use_cuda, device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed, gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)
    print('Set seeds to {}'.format(seed))


def diarization_experiment(
    model_args, training_args, inference_args, data_args
):
    """Experiment pipeline.

    Load data --> train model --> test model --> output result

    Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
    data_args: data configurations
    """
    start = datetime.now()

    if training_args.debug:
        print('\n\n===== DEBUG MODE =====\n\n')

    def debug(m):
        if training_args.debug:
            print(m)

    predicted_cluster_ids = []
    test_record = []

    # Reproducibility
    set_seed(training_args.seed)

    # Load data
    train_sequence = np.load(data_args.train_seq, allow_pickle=True)
    train_cluster_id = np.load(data_args.train_clusters, allow_pickle=True)
    test_sequence = np.load(data_args.test_seq, allow_pickle=True)
    test_cluster_id = np.load(data_args.test_clusters, allow_pickle=True)

    # Create model class
    model = uisrnn.UISRNN(model_args)
    print('{} - Created {} model with {:,} params:'.format(
        datetime.now() - start, model.__class__.__name__,
        count_parameters(model.rnn_model)
    ))
    print(model.rnn_model)

    # Training
    model_loc = os.path.join(training_args.out_dir, MODEL_NAME_STUB)
    model_constructed = (not training_args.overwrite) \
         and os.path.exists(model_loc)
    if model_constructed:
        try:
            model.load(model_loc)
            print('{} - Loaded trained model from {}'.format(
                datetime.now() - start, model_loc,
            ))
        except Exception as e:
            print('Unable to load model from {}:\n{}'.format(
                model_loc, e
            ))
            model_constructed = False
    if not model_constructed:
        model.fit(train_sequence, train_cluster_id, training_args)
        print('{} - Trained model!'.format(datetime.now() - start))
        model.save(model_loc)
        print('{} - Saved model to {}'.format(
            datetime.now() - start, model_loc
        ))

    # Testing
    with torch.no_grad():
        for test_seq, test_cluster in zip(
            test_sequence, test_cluster_id
        ):
            debug('Test seq shape: {}'.format(test_seq.shape))
            debug('Test cluster: {}'.format(test_cluster))
            predicted_cluster_id = model.predict(test_seq, inference_args)
            debug('Predicted cluster ID: {}, class {}'.format(
                predicted_cluster_id,
                predicted_cluster_id.__class__.__name__
            ))
            predicted_cluster_ids.append(predicted_cluster_id)
            accuracy = uisrnn.compute_sequence_match_accuracy(
                test_cluster, predicted_cluster_id)
            test_record.append((accuracy, len(test_cluster)))
            print('Ground truth labels:')
            print(test_cluster)
            print('Predicted labels:')
            print(predicted_cluster_id)
            print('-' * 80)

    output_string = uisrnn.output_result(
        model_args,
        training_args,
        test_record
    )

    print('Finished diarization experiment')
    print(output_string)

def main():
    """The main function."""
    # Retrieve arguments
    model_args, training_args, \
        inference_args, data_args = uisrnn.parse_arguments()

    # Run experiment
    diarization_experiment(
        model_args,
        training_args,
        inference_args,
        data_args
    )

"""
==== TIMIT ====
python3 -u train.py --enable-cuda --batch_size 50 \
--out-dir /data4/shuyang/TIMIT_spk \
--train-seq /data4/shuyang/TIMIT_spk/TRAIN_sequence.npy \
--train-clusters /data4/shuyang/TIMIT_spk/TRAIN_cluster_id.npy \
--test-seq /data4/shuyang/TIMIT_spk/TEST_sequence.npy \
--test-clusters /data4/shuyang/TIMIT_spk/TEST_cluster_id.npy

"""
if __name__ == '__main__':
    main()
