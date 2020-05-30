import torch
import torch.nn as nn
from mv_lstm import MV_LSTM
from data_generator import SlidingWindow
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

# Training
def train(data_loader_train, input_size, num_classes, seq_len, data_loader_val=None, num_epochs=100, device="cpu"):
    learning_rate = 1e-3
    hidden_size = 256
    num_layers = 1

    lstm = MV_LSTM(input_size=input_size,
                   seq_length=seq_len,
                   num_output=num_classes,
                   n_hidden=hidden_size,
                   n_layers=num_layers)
    lstm.to(device)

    criterion = torch.nn.MSELoss().to(device)  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-7, eps=1e-08)

    # Not it probably won't work with other batch size
    batch_size = 1

    for epoch in range(num_epochs):
        # idea - created iterative data generator instead?
        lstm.train()
        for x_batch, y_batch in data_loader_train:
            lstm.zero_grad()
            lstm.init_hidden(x_batch.size(0))

            output = lstm(x_batch.to(device))
            # print(output.shape)
            # print(y_batch.shape)
            loss = criterion(output, y_batch.to(device))
            loss.backward()

            optimizer.step()
            del x_batch
            del y_batch

        vall_loss = None
        if data_loader_val is not None:
            with torch.no_grad():
                # Evaluate on test
                for x_batch_val, y_batch_val in data_loader_val:
                    lstm.eval()
                    lstm.init_hidden(x_batch_val.size(0))
                    valid = lstm(x_batch_val.to(device))
                    vall_loss = criterion(valid, y_batch_val.to(device))
                    scheduler.step(vall_loss)

        if (epoch + 1) % 25 == 1:
            if vall_loss is None:
                print("Epoch: %d, loss: %1.5f " % (epoch, loss.cpu().item()))
            else:
                print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " % (epoch, loss.cpu().item(), vall_loss.cpu().item()))

    print("Epoch: %d, loss: %1.5f " % (epoch, loss.cpu().item()))
    return lstm.eval()


def main():
    # Demonstration of data generation
    # define input sequence
    in_seq1 = np.array([x for x in range(0, 1000, 10)])
    in_seq2 = np.array([x for x in range(5, 1005, 10)])
    in_seq3 = np.array([x for x in range(10, 1010, 10)])
    out_seq = np.array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    in_seq3 = in_seq3.reshape((len(in_seq3), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # horizontally stack columns
    dataset = np.hstack((in_seq1, in_seq2, in_seq3, out_seq))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(dataset)
    window = 10

    print(dataset)

    sliding_window_dataset = SlidingWindow(train_data_normalized, train_seq_len=window)
    data_loader = DataLoader(sliding_window_dataset, batch_size=10)
    model = train(data_loader,
                  input_size=sliding_window_dataset.input_size,
                  num_classes=sliding_window_dataset.num_classes,
                  seq_len=sliding_window_dataset.train_seq_len,
                  num_epochs=350)
    input = torch.from_numpy(np.array([train_data_normalized[-window:, :]])).float()
    model.init_hidden(input.size(0))
    prediction = model(input.to("cpu")).detach().numpy()
    print(prediction)
    prediction_transformed = scaler.inverse_transform(prediction)
    print(prediction_transformed)


if __name__ == '__main__':
    main()