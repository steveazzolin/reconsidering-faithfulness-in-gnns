import os
import argparse
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import random
import data_utils
from rage import RAGE,train,eval
from sklearn.model_selection import train_test_split

torch.set_num_threads(6)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Proteins')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate. Default is 0.0. ')
    parser.add_argument('--batch-size', type=int, default=256) # era 128
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GCN layers. Default is 3.')
    parser.add_argument('--dim', type=int, default=20, help='Number of GCN dimensions. Default is 20. ')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs. Default is 100. ')
    parser.add_argument('--inner_epoch', type=int, default=10) # era 20
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Default is 0.001. ')
    parser.add_argument('--device', type=int, default=0, help='Index of cuda device to use. Default is 0. ')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    #parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--method', type=str, default='classification')
    parser.add_argument('--explainer_layer', type=str, default='gin') # it was gcn
    parser.add_argument('--gnn_layer', type=str, default='gin') # it was gcn
    parser.add_argument('--gnn_pool', type=str, default='mean') # it was max #m2 for mitigation mean otherwise
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--start_run', type=int, default=1)
    parser.add_argument("--mitigation", type=str, default=None)

    return parser.parse_args()


def main(seed = 1):
    args = parse_args()
    args.random_seed = seed * 97 + 13

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Load and split the dataset.
    if args.dataset == "GOODMotif2" or args.dataset == "GOODMotif_size":
        dataset = data_utils.load_dataset(args.dataset)
        train_set = dataset["train"]
        valid_set = dataset["val"]
        test_set = dataset["test"]
        dataset = train_set
        print("train",np.mean([d.num_edges for d in train_set]))
        print("val",np.mean([d.num_edges for d in valid_set]))
        print("test",np.mean([d.num_edges for d in test_set]))

    else:
        dataset = data_utils.load_dataset(args.dataset)
        #split mio
        idx_train, idx_test = train_test_split(np.arange(len(dataset)),test_size= 0.2,random_state=42)
        idx_test, idx_val = train_test_split(idx_test, test_size= 0.5,random_state=42)    
        train_set = dataset[idx_train]
        valid_set = dataset[idx_val]
        test_set = dataset[idx_test]

    #split originale
    #splits, indices = data_utils.split_data(dataset)w
    #train_set, valid_set, test_set = splits

    args.num_features = dataset.num_features
    args.num_classes = train_set.num_classes
    print("num classes", args.num_classes)

    # Logging.
    result_folder = f'data/{args.dataset}/rage/'
    args.result_folder = result_folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    log_file = result_folder + f'log.txt'
    with open(log_file, 'w') as f:
        pass

    if args.dataset == 'Tree-of-Life':
        args.num_classes = 1
        args.method = 'regression'
        
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    train_scores = {'accuracy': [], 'auc': [], 'ap': [], 'mae': [], 'mse': [], 'r2': []}
    valid_scores = {'accuracy': [], 'auc': [], 'ap': [], 'mae': [], 'mse': [], 'r2': []}
    test_scores = {'accuracy': [], 'auc': [], 'ap': [], 'mae': [], 'mse': [], 'r2': []}
    
    run = args.random_seed

    args.run = run
    torch.backends.cudnn.deterministic = True
    random.seed(run)
    torch.manual_seed(run)
    torch.cuda.manual_seed(run)
    np.random.seed(run)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Initialize the model.    
    
    if args.gnn_pool == "m2":
        print("mitaigation = m2")
    elif args.mitigation == "HM":
        print("mitaigation = HM")
    elif args.mitigation == " m2HM":
        MITIGATION = "m2HM"
        print("both mitigations")
        args.gnn_pool == "m2"

    model = RAGE(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    # Training.
    start_epoch = 1
    #epoch_iterator = tqdm(range(start_epoch, start_epoch + args.epochs), desc='Epoch')
    best_valid = float('inf')
    best_epoch = 0
    patience = int(args.epochs / 5)
    cur_patience = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):

        train_loss = train(model, optimizer, train_loader, valid_loader, device, method=args.method, args=args)
        valid_loss = eval(model, valid_loader, device, method=args.method, args=args)
        if valid_loss[0] < best_valid:
            cur_patience = 0
            best_valid = valid_loss[0]
            if args.gnn_pool == "m2":
                torch.save(model.state_dict(), result_folder + f'best_model_run_{run}_m2.pt')
            elif args.mitigation == "HM":
                torch.save(model.state_dict(), result_folder + f'best_model_run_{run}_HM.pt')
            elif args.mitigation == "m2HM":
                torch.save(model.state_dict(), result_folder + f'best_model_run_{run}_m2HM.pt')
            else:
                torch.save(model.state_dict(), result_folder + f'best_model_run_{run}.pt')
        else:
            cur_patience += 1
            if cur_patience >= patience:
                break

        if epoch % 10 == 0:             
            test_loss = eval(model, test_loader, device, method=args.method, args=args)
            print(epoch,"train_loss: %.2f val_loss: %.2f val_acc: %.2f test loss: %.2f test_acc: %.2f " %(train_loss,valid_loss[0],valid_loss[3],test_loss[0],test_loss[3]))

    # Testing.
    if args.gnn_pool == "m2":
        model.load_state_dict(torch.load(result_folder + f'best_model_run_{run}_m2.pt', map_location=device))
    elif args.mitigation == "HM":
        model.load_state_dict(torch.load(result_folder + f'best_model_run_{run}_HM.pt', map_location=device))
    elif args.mitigation == "m2HM":
        model.load_state_dict(torch.load(result_folder + f'best_model_run_{run}_m2HM.pt', map_location=device))
    else:
        model.load_state_dict(torch.load(result_folder + f'best_model_run_{run}.pt', map_location=device))

    # evaluation
    if args.method == 'classification':
        train_loss, train_auc, train_ap, train_acc, train_preds, train_grounds = eval(model, train_loader, device, method=args.method, args=args)
        valid_loss, valid_auc, valid_ap, valid_acc, valid_preds, valid_grounds = eval(model, valid_loader, device, method=args.method, args=args)
        test_loss, test_auc, test_ap, test_acc, test_preds, test_grounds = eval(model, test_loader, device, method=args.method, args=args)
    else:
        train_loss, train_r2, train_mse, train_mae, train_preds, train_grounds = eval(model, train_loader, device, method=args.method, args=args)
        valid_loss, valid_r2, valid_mse, valid_mae, valid_preds, valid_grounds = eval(model, valid_loader, device, method=args.method, args=args)
        test_loss, test_r2, test_mse, test_mae, test_preds, test_grounds = eval(model, test_loader, device, method=args.method, args=args)

    torch.save((train_preds, train_grounds), result_folder + f'train_predictions_run_{run}.pt')
    torch.save((valid_preds, valid_grounds), result_folder + f'valid_predictions_run_{run}.pt')
    torch.save((test_preds, test_grounds), result_folder + f'test_predictions_run_{run}.pt')

    if args.method == 'classification':
        train_scores['auc'].append(train_auc)
        train_scores['ap'].append(train_ap)
        train_scores['accuracy'].append(train_acc)
        valid_scores['auc'].append(valid_auc)
        valid_scores['ap'].append(valid_ap)
        valid_scores['accuracy'].append(valid_acc)
        test_scores['auc'].append(test_auc)
        test_scores['ap'].append(test_ap)
        test_scores['accuracy'].append(test_acc)
    else:
        train_scores['r2'].append(train_r2)
        train_scores['mse'].append(train_mse)
        train_scores['mae'].append(train_mae)
        valid_scores['r2'].append(valid_r2)
        valid_scores['mse'].append(valid_mse)
        valid_scores['mae'].append(valid_mae)
        test_scores['r2'].append(test_r2)
        test_scores['mse'].append(test_mse)
        test_scores['mae'].append(test_mae)

    print(f"Train Scores = {train_scores}", flush=True)
    print(f"Valid Scores = {valid_scores}", flush=True)
    print(f"Test Scores = {test_scores}", flush=True)
    all_scores = {'train': train_scores, 'valid': valid_scores, 'test_scores': test_scores}
    torch.save(all_scores, result_folder + f'all_scores_{args.start_run}_{args.runs + args.start_run - 1}.pt')
    with open(log_file, 'a') as f:
        print(file=f)
        print(f"Train Scores = {train_scores}", file=f)
        print(f"Valid Scores = {valid_scores}", file=f)
        print(f"Test Scores = {test_scores}", file=f)

        if args.method == 'classification':
            print(f"Train AUC = {np.mean(train_scores['auc'])} +- {np.std(train_scores['auc'])}", file=f)
            print(f"Valid AUC = {np.mean(valid_scores['auc'])} +- {np.std(valid_scores['auc'])}", file=f)
            print(f"Test AUC = {np.round(np.mean(test_scores['auc']), 4)} +- {np.round(np.std(test_scores['auc']), 4)}", file=f)

            print(f"Train AP = {np.mean(train_scores['ap'])} +- {np.std(train_scores['ap'])}", file=f)
            print(f"Valid AP = {np.mean(valid_scores['ap'])} +- {np.std(valid_scores['ap'])}", file=f)
            print(f"Test AP = {np.round(np.mean(test_scores['ap']), 4)} +- {np.round(np.std(test_scores['ap']), 4)}", file=f)

            print(f"Train Accuracy = {np.mean(train_scores['accuracy'])} +- {np.std(train_scores['accuracy'])}", file=f)
            print(f"Valid Accuracy = {np.mean(valid_scores['accuracy'])} +- {np.std(valid_scores['accuracy'])}", file=f)
            print(f"Test Accuracy = {np.round(np.mean(test_scores['accuracy']), 4)} +- {np.round(np.std(test_scores['accuracy']), 4)}", file=f)
        else:
            print(f"Train R2 = {np.mean(train_scores['r2'])} +- {np.std(train_scores['r2'])}", file=f)
            print(f"Valid R2 = {np.mean(valid_scores['r2'])} +- {np.std(valid_scores['r2'])}", file=f)
            print(f"Test R2 = {np.round(np.mean(test_scores['r2']), 4)} +- {np.round(np.std(test_scores['r2']), 4)}", file=f)

            print(f"Train MSE = {np.mean(train_scores['mse'])} +- {np.std(train_scores['mse'])}", file=f)
            print(f"Valid MSE = {np.mean(valid_scores['mse'])} +- {np.std(valid_scores['mse'])}", file=f)
            print(f"Test MSE = {np.round(np.mean(test_scores['mse']), 4)} +- {np.round(np.std(test_scores['mse']), 4)}", file=f)

            print(f"Train MAE = {np.mean(train_scores['mae'])} +- {np.std(train_scores['mae'])}", file=f)
            print(f"Valid MAE = {np.mean(valid_scores['mae'])} +- {np.std(valid_scores['mae'])}", file=f)
            print(f"Test MAE = {np.round(np.mean(test_scores['mae']), 4)} +- {np.round(np.std(test_scores['mae']), 4)}", file=f)


if __name__ == '__main__':

    main(seed=1)
    main(seed=2)
    main(seed=3)
    main(seed=4)
    main(seed=5)
