from torch.utils.tensorboard.writer import SummaryWriter
writer = SummaryWriter(log_dir="runs/kdd99/memo/kdd_fc_128/Accuracy_curve")
writer.add_scalar("Accuracy_Curve", 100, 0)
writer.add_scalar("Accuracy_Curve", 99.71, 1)
writer.add_scalar("Accuracy_Curve", 99.9, 2)
writer.add_scalar("Accuracy_Curve", 99.99, 3)
writer.add_scalar("Accuracy_Curve", 99.06, 4)
writer.add_scalar("Accuracy_Curve", 99.56, 5)
writer.close()
