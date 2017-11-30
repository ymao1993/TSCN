""" Script for converting the evaluation result into the submission format
of Activity-Net Dense Caption using the ground-truth proposals.
"""
import json
import argparse
import os

ground_truth_proposals_dump = 'format_convert/valdump.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder')
    parser.add_argument('output_folder')
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    assert os.path.exists(input_folder), 'input_folder %s does not exist.' % input_folder
    assert os.path.exists(output_folder), 'output_folder %s does not exist.' % output_folder

    # Read the ground_truth_proposal_dump.
    ground_truth_proposals = json.load(open(ground_truth_proposals_dump))
    valmap = ground_truth_proposals['valmap']
    valtimes = ground_truth_proposals['valtimes']

    input_files = os.listdir(input_folder)
    for input_file in input_files:
        if not input_file.endswith('.json'):
            print('Skip %s' % input_file)
            continue

        print('Converting %s...' % input_file)
        input_file_path = os.path.join(input_folder, input_file)
        output_file_path = os.path.join(output_folder, input_file)

        # Read evaluation result json file and convert it to submission format.
        results = {}
        data = json.load(open(input_file_path))
        print(input_file_path)
        for i in range(len(data)):
            video_segment_result = data[i]
            video_segment_name = video_segment_result['name']
            sample = {
                'sentence': video_segment_result['video_caption'],
                'timestamp': valtimes[video_segment_name]
            }
            video_name = valmap[video_segment_name]
            if video_name not in results:
                results[video_name] = [sample]
            else:
                results[video_name].append(sample)

        # Dump final result.
        final_json_dump = {
            'version': 'VERSION 1.0',
            'external_data': 'NULL',
            'results': results
        }
        json.dump(final_json_dump, open(output_file_path, 'w'))


if __name__ == '__main__':
    main()
