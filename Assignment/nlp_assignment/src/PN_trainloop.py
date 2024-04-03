import torch



def train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        total_correct += torch.sum(preds == labels).item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    return average_loss, accuracy

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            total_correct += torch.sum(preds == labels).item()

    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    return average_loss, accuracy
