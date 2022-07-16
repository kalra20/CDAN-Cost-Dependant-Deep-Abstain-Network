import argparse

from models.CDAN_cifar import CIFARVgg
from CDAN_utils import *

MODELS = {"cifar_10": CIFARVgg}
# "fmnist": FMNIST
def to_train(filename):
    checkpoints = os.listdir("history_checkpoints/")
    if filename in checkpoints:
        return False
    else:
        return True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar_10')

# parser.add_argument('--model_name', type=str, default='test')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--mu',type=float, default=1.0)
parser.add_argument('--filename',type=str,default='CDAN_cifar10.h5')
parser.add_argument('--epochs',type=int,default=250)
parser.add_argument('--lr',type=float,default=1e-2)
args = parser.parse_args()

model_cls = MODELS[args.dataset]
# model_name = args.model_name
# baseline_name = args.baseline

cost_reject = [ 0.4, 0.35, 0.3, 0.25]

for cost in cost_reject:
    # filename = args.dataset+str(cost)+'.h5'
    
    model = model_cls(train=to_train("{}_{}.h5".format(args.dataset, cost)), filename="{}_{}.h5".format(args.dataset, cost), alpha=args.alpha,d=cost,mu= args.mu,epochs=args.epochs,lr=args.lr)