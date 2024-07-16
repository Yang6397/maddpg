# encoding: utf-8
# desc: 超参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--max_epochs', type=int, default=1000, help='max_epochs')
parser.add_argument('--max_steps', type=int, default=200, help='max_steps')
parser.add_argument('--memory_size', type=int, default=1024, help='memory_size')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
parser.add_argument('--update_interval', type=int, default=100, help='update_interval')
parser.add_argument('--lr_actor', type=float, default=0.01, help='lr_actor')
parser.add_argument('--lr_critic', type=float, default=0.01, help='lr_actor')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--tau', type=float, default=0.01, help='tau')
args = parser.parse_args()
