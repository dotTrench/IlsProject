from pandas import read_csv
import scipy.stats
from operator import itemgetter


def main():
    data = read_csv('dataset/train/train.csv')
    out = []
    for column in data.columns[:-1]:
        x = data[column]
        y = data['recieved_pizza(class)']
        out.append({
            'name': column,
            'pearson': scipy.stats.spearmanr(x, y)[0],
            'spearman': scipy.stats.pearsonr(x, y)[0]
            })

    print('Pearson: ')
    for o in sorted(out, key=itemgetter('pearson'), reverse=True):
        print('{0} {1}'.format(o['name'], o['pearson']))

    print('------')
    print('Spearman: ')

    for o in sorted(out, key=itemgetter('spearman'), reverse=True):
        print('{0} {1}'.format(o['name'], o['spearman']))

if __name__ == '__main__':
    main()
