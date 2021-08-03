import json


def main():
    data = json.load(open('local/speechocean762/scores.json'))
    res = {}
    for utt, d in data.items():
        exist = False
        for word in d['words']:
            for acc in word['phones-accuracy']:
                if round(acc) == 0:
                    exist = True
                    break

            if exist:
                break

        if exist:
            res[utt] = d

    json.dump(res, open('local/speechocean762/score_samples.json', 'w'))


if __name__ == '__main__':
    main()
