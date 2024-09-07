def tape_upsampling(cams_pose_list):
    upsampling = []
    idx = 0
    for i in range(len(cams_pose_list) - 1):
        current = cams_pose_list[i]
        next_point = cams_pose_list[i + 1]
        current['id'] = idx
        upsampling.append(current)
        idx = idx+1

        diff = [(b - a) / 5 for a, b in zip(current['position'], next_point['position'])]
        # upsampling from 10hz to 50hz:
        for j in range(1, 5):
            new_pos = [current['position'][k] + j * diff[k] for k in range(3)]
            new_id = idx
            upsampling.append({'id': new_id,
                               'position': new_pos})
            idx = idx+1

    # 添加最后一个点
    cams_pose_list[-1]['id'] = idx
    upsampling.append(cams_pose_list[-1])

    return upsampling


cp = [{'id': 0, 'position': [1, 1, 3]}, {'id': 1, 'position': [2, 2, 4]}, {'id': 2, 'position': [5, 5, 5]}]

output = tape_upsampling(cp)

print(output)