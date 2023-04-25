import random
import typing
import torch
from timeit import default_timer as timer

import torchmetrics
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm
from enum import Enum


def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    total_time = end - start
    print(f"Total time on {device}: {total_time:.3f} seconds")
    return total_time


def eval_model(model: torch.nn.Module, #TODO: убрать за ненадобностью
               data_loader: torch.utils.data.DataLoader,
               loss_fn: typing.Callable,
               eval_fn,
               device: torch.device,
               train_time: float = 0
               ):
    loss, eval = 0, 0
    model.eval()
    y_predicted = []
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            y_predicted.append(y_pred.argmax(dim=1))
            loss += loss_fn(y_pred, y)
            eval += eval_fn(y, y_pred.argmax(dim=1))

        loss /= len(data_loader)
        eval /= len(data_loader)
        y_predicted = torch.cat(y_predicted, dim=0)
    print(y_predicted.shape)
    return {
        "name": model.__class__.__name__,
        "loss": loss.item(),
        "eval": eval.item(),
        "train_time": train_time,
        "predicted": torch.flatten(y_predicted)
    }


def model_eval_report(
        model: torch.nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        test_dataset: torch.utils.data.Dataset,
        loss_function: typing.Callable,
        eval_function: typing.Callable,
        device: torch.device,
        transformer=False,
        class_names=[]
):
    # eval_res = eval_model(
    #     model=model,
    #     data_loader=test_dataloader,
    #     loss_fn=loss_function,
    #     eval_fn=eval_function,
    #     device=device,
    #     train_time=0,
    # )
    _, _, preds = test_step(
        model=model,
        dataloader=test_dataloader,
        loss_function=loss_function,
        eval_function=eval_function,
        device=device,
        transformer=transformer
    )
    # TODO: костыль
    targets = torch.tensor(test_dataset.targets if not transformer else test_dataset.tensors[2])
    targets = targets.to(device)
    class_names = test_dataset.classes if not transformer else class_names
    confmat = torchmetrics.ConfusionMatrix(num_classes=len(class_names), task='multiclass').to(device)

    confmat_tensor = confmat(preds=preds,
                             target=targets).to(device)

    plot_confusion_matrix(
        conf_mat=confmat_tensor.cpu().numpy(),
        class_names=class_names,
        figsize=(7, 5)
    )

    # print(eval_res)
    print(classification_report(
        y_pred=preds.cpu().numpy(),
        y_true=targets.cpu().numpy(),
        target_names=class_names
    ))


class TrainStatus(Enum):
    NONE = 0
    Success = 1
    GradientExplosion = 2


def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_function: typing.Callable,
        optimizer: torch.optim.Optimizer,
        eval_function: typing.Callable,
        device: torch.device,
        transformer=False
) -> (float, float, TrainStatus, int):
    """
    Train step for model: calc loss, zero_grad, backward, step

    :return: loss, acc, train status, on which batch broken
    """

    train_loss, train_eval = 0, 0
    status = TrainStatus.NONE
    batch_step_save = 0
    model.train()

    if transformer:
        for i, (input_ids, input_mask, labels) in enumerate(tqdm(dataloader, desc="Train batches")):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)

            model_res = model(input_ids,
                              token_type_ids=None,
                              attention_mask=input_mask,
                              labels=labels)

            if torch.isnan(model_res["logits"]).any():
                status = TrainStatus.GradientExplosion
                batch_step_save = i
                break

            # loss = loss_function(y_train_pred, labels)
            train_loss += model_res['loss']

            train_eval += eval_function(labels, model_res['logits'].argmax(dim=1))

            optimizer.zero_grad()
            model_res['loss'].backward()
            optimizer.step()
    else:
        for i, (X_train, y_train) in enumerate(tqdm(dataloader, desc="Train batches")):
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            y_train_pred = model(X_train)

            if torch.isnan(y_train_pred).any():
                status = TrainStatus.GradientExplosion
                batch_step_save = i
                break

            loss = loss_function(y_train_pred, y_train)
            train_loss += loss

            train_eval += eval_function(y_train, y_train_pred.argmax(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_loss /= len(dataloader)
    train_eval /= len(dataloader)

    if status is TrainStatus.NONE:
        status = TrainStatus.Success
    else:
        train_loss, train_eval = (torch.tensor([0]),) * 2

    return train_loss, train_eval, status, batch_step_save


def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_function: typing.Callable,
        eval_function: typing.Callable,
        device: torch.device,
        transformer=False
) -> (float, float):
    """
    :return: loss, acc
    """
    model.eval()
    test_eval, test_loss = 0, 0
    y_predicted = []
    with torch.inference_mode():
        if transformer:
            for i, (input_ids, input_mask, labels) in enumerate(tqdm(dataloader, desc="Train batches")):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                labels = labels.to(device)

                y_test_pred = model(input_ids,
                                    token_type_ids=None,
                                    attention_mask=input_mask,
                                    labels=labels)
                y_predicted.append(y_test_pred['logits'].argmax(dim=1))
                test_loss += y_test_pred['loss']
                test_eval += eval_function(labels, y_test_pred['logits'].argmax(dim=1))
        else:
            for X_test, y_test in tqdm(dataloader, desc="Test batches"):
                X_test, y_test = X_test.to(device), y_test.to(device)
                y_test_pred = model(X_test)
                y_predicted.append(y_test_pred.argmax(dim=1))
                test_loss += loss_function(y_test_pred, y_test)
                test_eval += eval_function(y_test, y_test_pred.argmax(dim=1))
        test_loss /= len(dataloader)
        test_eval /= len(dataloader)
        y_predicted = torch.cat(y_predicted, dim=0)
    return test_loss, test_eval, y_predicted


def get_name_of_obj_or_fn(fn_or_obj) -> typing.AnyStr:
    """
    In case various types can be passed as callable
    :param fn_or_obj: Function | Instantiated object
    :return: String - name
    """
    return fn_or_obj.__name__ if hasattr(fn_or_obj, '__name__') else fn_or_obj.__class__.__name__


def model_trainer(
        model: torch.nn.Module,
        epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_function: typing.Callable,
        optimizer: torch.optim.Optimizer,
        eval_function: typing.Callable,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        transformer=False
) -> typing.Mapping[typing.AnyStr, float or typing.AnyStr or typing.Collection]:
    """
    Helper function to prevent writing each.
    MODEL MUST HAVE name FIELD

    :return: Model training report
    """
    train_time_start = timer()
    model.to(device)

    train_loss, train_eval, test_loss, test_eval = (torch.tensor([0]),) * 4
    history_train_loss, history_train_eval, history_test_loss, history_test_eval = [], [], [], []
    train_status = TrainStatus.NONE
    epoch_save = 0
    batch_save = None
    print(f"---------------------\nIterating on {model.name}, LR = {optimizer.param_groups[0]['lr']}")
    for epoch in tqdm(range(epochs), desc=f"Epochs of {model.name}"):
        print(f"Epoch {epoch + 1}\n-------")
        epoch_save = epoch + 1
        train_loss, train_eval, train_status, breaking_batch = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            eval_function=eval_function,
            device=device,
            transformer=transformer
        )

        history_train_loss.append(round(train_loss.item(), 5))
        history_train_eval.append(round(train_eval.item(), 5))

        if train_status is TrainStatus.GradientExplosion:
            batch_save = breaking_batch
            break

        print(f"---Train--- \n{get_name_of_obj_or_fn(eval_function)}: {train_eval * 100}\nLoss: {train_loss}\n")

        test_loss, test_eval, _ = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_function=loss_function,
            eval_function=eval_function,
            device=device,
            transformer=transformer
        )

        history_test_loss.append(round(test_loss.item(), 5))
        history_test_eval.append(round(test_eval.item(), 5))

        print(f"---Test--- \n{get_name_of_obj_or_fn(eval_function)}: {test_eval * 100}\nLoss: {test_loss}\n")
        if scheduler is not None:
            scheduler.step()

    train_time_end = timer()
    return {
        # Train status info
        "name": model.name,
        "class_name": model.__class__.__name__,
        "status": train_status,
        "epochs_remain": epochs - epoch_save,
        "on_which_batch_broken": batch_save,

        # General info
        "train_time": print_train_time(
            start=train_time_start,
            end=train_time_end,
            device=device
        ),
        "loss_function": get_name_of_obj_or_fn(loss_function),
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],  # TODO : can be other groups
        "eval_function": get_name_of_obj_or_fn(eval_function),
        "device": device,

        # Final scores
        "end_test_loss": test_loss.item(),
        "end_test_eval": test_eval.item(),
        "end_train_loss": train_loss.item(),
        "end_train_eval": train_eval.item(),

        # History of scores changing
        "history_test_loss": history_test_loss,
        "history_test_eval": history_test_eval,
        "history_train_loss": history_train_loss,
        "history_train_eval": history_train_eval,

    }


def make_prediction(
        model: torch.nn.Module,
        data: list,
        device: torch.device
):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)


def plot_random_samples(
        test_data: torch.utils.data.Dataset,
        model: torch.nn.Module,
        device: torch.device,
        class_names: list,
        figsize: int,
        rows_cols: int,
        seed: int = -1
):
    if seed != -1:
        random.seed(seed)
    test_samples = []
    test_labels = []

    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = make_prediction(model=model, data=test_samples, device=device)
    pred_classes = pred_probs.argmax(dim=1)

    plt.figure(figsize=(figsize, figsize))
    rows, cols = rows_cols, rows_cols
    for i, sample in enumerate(test_samples):
        plt.subplot(rows, cols, i + 1)
        if sample.shape[0] == 3:
            plt.imshow(sample.permute(1, 2, 0))
        else:
            plt.imshow(sample.squeeze())
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]
        title_wrong_text = f"Pred {pred_label} | True: {truth_label}"
        title_correct_text = pred_label
        if pred_label == truth_label:
            plt.title(title_correct_text, fontsize=10, c="g")
        else:
            plt.title(title_wrong_text, fontsize=10, c="r")
        plt.axis(False)
