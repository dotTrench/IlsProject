import json


def word_count_convert(value):
    return len(value)


fields = [
    {
        'input': 'requester_account_age_in_days_at_request',
        'output': 'requester_account_age'
    },
    {
        'input': 'request_text_edit_aware',
        'converter': word_count_convert,
        'output': 'request_text_length'
    },
    {
        'input': 'request_title',
        'converter': word_count_convert,
        'output': 'request_title_length'
    },
    {
        'input': 'requester_number_of_comments_at_request',
        'output': 'requester_num_comments'
    },
    {
        'input': 'requester_number_of_comments_in_raop_at_request',
        'output': 'requester_num_comments_in_raop'
    },
    {
        'input': 'requester_number_of_posts_at_request',
        'output': 'requester_num_posts'
    },
    {
        'input': 'requester_number_of_posts_on_raop_at_request',
        'output': 'requester_num_posts_in_raop'
    },
    {
        'input': 'requester_number_of_subreddits_at_request',
        'output': 'requester_num_subreddits'
    },
    {
        'input': 'requester_upvotes_minus_downvotes_at_request',
        'output': 'requester_upvotes_minus_downvotes'
    },
    {
        'input': 'requester_username',
        'converter': word_count_convert,
        'output': 'requester_username_length'
    },
    # This has to be the last field since it's the class column!
    {
        'input': 'requester_received_pizza',
        'output': 'recieved_pizza(class)'
    }
]


def read_input_file(filepath):
    json_file = open(filepath)
    data = json.loads(json_file.read())
    json_file.close()
    return data


def convert_entry_to_row(entry):
    values = []
    for field in fields:
        value = entry[field['input']]

        output_value = None
        if 'converter' in field:
            output_value = field['converter'](value)
        else:
            output_value = value

        values.append(str(output_value))

    return ','.join(values)


def get_headers():
    return ','.join([field['output'] for field in fields])


def main():
    print('Reading JSON input')
    entries = read_input_file('dataset/train/train.json')
    out_file = open('dataset/train/train.csv', 'w')
    headers = get_headers()

    print('Writing headers')
    out_file.write('{}\n'.format(headers))

    print('Converting input')
    rows = [convert_entry_to_row(entry) for entry in entries]

    print('Writing to file')
    for row in rows:
        out_file.write('{0}\n'.format(row))

    print('Closing file')
    out_file.close()

    print('Done')


if __name__ == '__main__':
    main()
