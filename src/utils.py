def getTestAccuracy(model, dataset, features=None):
  X, y = dataset.tensors
  if features is not None:
    filter = torch.zeros(X.shape[1])
    filter[features.copy()] = 1
    X = torch.mul(X, filter)

  y = y.cpu().detach().numpy()

  if type(model) == sklearn.svm._classes.LinearSVC:
    X = X.cpu().detach().numpy()
    y_pred = model.predict(X)
  else:
    with torch.no_grad():
      model.eval()
      y_pred = torch.softmax(model(X), dim=1).cpu().detach().numpy().argmax(axis=1)
  accuracy = (y==y_pred).sum()/y.shape[0]
  per_class_accuracy = confusion_matrix(y_pred, y, normalize='true').diagonal()
  return accuracy, per_class_accuracy.mean()


def getLogs(folder, find_version=True):

  if find_version:
    versions = [f for f in os.listdir(folder) if f.startswith('version')]
    versions.sort()
    folder = os.path.join(folder, versions[-1])

  print("Reading logs from %s..." % folder)

  logs = pd.read_csv(os.path.join(folder, "metrics.csv"))
  logs.columns = [col.split('/')[0] for col in logs.columns]

  logs_step = logs[logs['train_batch_loss'].notnull()].dropna(axis=1)
  logs_step.set_index('step', inplace=True)
  logs_epoch = logs[logs['val_eval_loss'].notnull()].dropna(axis=1)
  logs_epoch.set_index('epoch', inplace=True)

  return logs_step, logs_epoch