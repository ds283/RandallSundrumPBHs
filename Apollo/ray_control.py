import argparse
import pandas as pd
import fabric

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="SGE $PE_HOSTFILE to process", type=str, required=True)
parser.add_argument("--start", help="Set up Ray cluster", type=bool, default=None, action=argparse.BooleanOptionalAction)
parser.add_argument("--stop", help="Tear down Ray cluster", type=bool, default=None, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

df = pd.read_csv(args.file, header=None, delim_whitespace=True)

hosts = df.iloc[:,0]
slots = df.iloc[:,1]

num_rows = len(df.index)

if num_rows < 1:
    raise RuntimeError('Too few hosts specified (num hosts={n}'.format(n=num_rows))

# initiate ray head instance on first node
head_addr = hosts.iloc[0]
head_slots = slots.iloc[0]

if args.start:
    print('** Allocated head node "{name}", slots={slots}'.format(name=head_addr, slots=head_slots))
    head = fabric.Connection(host=head_addr, user='ds283').run('source /mnt/pact/ds283/anaconda3/etc/profile.d/conda.sh && /mnt/pact/ds283/anaconda3/envs/ray/bin/ray start --head --num-cpus {n}'.format(n=head_slots))
elif args.stop:
    head = fabric.Connection(host=head_addr, user='ds283').run('source /mnt/pact/ds283/anaconda3/etc/profile.d/conda.sh && /mnt/pact/ds283/anaconda3/envs/ray/bin/ray stop')


workers = []

if num_rows > 1:
    for row in range(1, num_rows):
        if args.start:
            print('** Allocated worker node "{name}", slots={slots}'.format(name=hosts.iloc[row], slots=slots.iloc[row]))
            workers.append(fabric.Connection(host=hosts.iloc[row], user='ds283').run('source /mnt/pact/ds283/anaconda3/etc/profile.d/conda.sh && /mnt/pact/ds283/anaconda3/envs/ray/bin/ray start --address {head_addr}:6379 --num-cpus {n}'.format(head_addr=head_addr, n=slots.iloc[row])))
        elif args.stop:
            workers.append(fabric.Connection(host=hosts.iloc[row], user='ds283').run('source /mnt/pact/ds283/anaconda3/etc/profile.d/conda.sh && /mnt/pact/ds283/anaconda3/envs/ray/bin/ray stop'))
