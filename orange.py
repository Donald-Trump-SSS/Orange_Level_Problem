import json
from collections import defaultdict

def parse_access_log(filepath):

    file_access_map = defaultdict(set)

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or ':' not in line:
                continue  

            username, filename = line.split(':', 1)
            username, filename = username.strip(), filename.strip()

            if not username or not filename:
                continue  

            file_access_map[filename].add(username)

    return dict(file_access_map)


def calculate_access_frequency(file_access_map):

    return {filename: len(users) for filename, users in file_access_map.items()}


def invert_access_mapping(file_access_map):

    user_access_map = defaultdict(set)
    for filename, users in file_access_map.items():
        for user in users:
            user_access_map[user].add(filename)
    return dict(user_access_map)


def summarize_access_log(filepath):
    file_to_users = parse_access_log(filepath)
    file_access_counts = calculate_access_frequency(file_to_users)
    user_to_files = invert_access_mapping(file_to_users)

    result = {
        "file_to_users": {f: list(u) for f, u in file_to_users.items()},
        "file_access_counts": file_access_counts,
        "user_to_files": {u: list(f) for u, f in user_to_files.items()}
    }

    return result


if __name__ == "__main__":
    summary = summarize_access_log('alicedata.log')

    print(json.dumps(summary, indent=4))
