""" This script select the best captions from multiple generated caption files.
"""
import argparse
import json
import nltk
import glob


def compute_bleu_score(caption, ref_caption):
    similarity_func = nltk.translate.bleu_score.sentence_bleu
    caption_tokens = nltk.word_tokenize(caption.lower())
    ref_caption_tokens = nltk.word_tokenize(ref_caption.lower())
    score = similarity_func([ref_caption_tokens], caption_tokens)
    return score


class CaptionResult:
    def __init__(self):
        self.name = ""
        self.gt_caption = ""
        self.candidate_captions = []
        self.best_caption = ""


def load_captioning_results(file_path, results):
    data = json.load(open(file_path,'r'))
    for entry in data:
        name = entry['name']
        gt_caption = entry['gt_caption']
        generated_caption = entry['video_caption']
        if name not in results:
            caption_result = CaptionResult()
            results[name] = caption_result
        else:
            caption_result = results[name]
        assert caption_result.gt_caption == gt_caption or caption_result.gt_caption == ""
        caption_result.gt_caption = gt_caption
        caption_result.candidate_captions.append(generated_caption)


def select_best_caption(results):
    count = 0
    total_count = len(results)
    for name, result in results.iteritems():
        count += 1
        print('Processing [%d/%d]%s...' % (count, total_count, name))
        candidate_captions = result.candidate_captions
        gt_caption = result.gt_caption
        assert len(candidate_captions) != 0
        max_score = -1
        best_candidate = ""
        for candidate_caption in candidate_captions:
            score = compute_bleu_score(candidate_caption, gt_caption)
            if max_score < score:
                best_candidate = candidate_caption
                max_score = score
        result.best_caption = best_candidate


def dump_results(results, file_path):
    json_dump = []
    for name, result in results.iteritems():
        json_dump.append({
            'name': name,
            'gt_caption': result.gt_caption,
            'video_caption': result.best_caption
        })
    fo = open(file_path, 'w')
    json.dump(json_dump, fo, indent=4)
    fo.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name_pattern')
    parser.add_argument('output_file_path')
    args = parser.parse_args()
    name_pattern = args.name_pattern
    output_file_path = args.output_file_path
    files = glob.glob(name_pattern)

    results = {}
    for file in files:
        load_captioning_results(file, results)
    select_best_caption(results)
    dump_results(results, output_file_path)

if __name__ == '__main__':
    main()
