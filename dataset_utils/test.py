import os


if __name__ == "__main__":
    option3_train = open('../datasets/option3/train.source')
    option3_valid = open('../datasets/option3/valid.source')

    merged_content_source = ""
    break_flag = 0
    while break_flag != 1:
        s1_flag = 0

        # Adding one line from source 1
        fps1_line = option3_valid.readline()
        if len(fps1_line) != 0:
            merged_content_source += fps1_line.split("\n")[0] + " </s>" + "\n"
        else:
            # Done reading the source 1
            s1_flag = 1
        if s1_flag == 1:
            break_flag = 1

    print(merged_content_source)

    output_dir1 = "../datasets/option3"
    with open(os.path.join(output_dir1, "valid.source"), "w") as f:
        f.write(merged_content_source)







