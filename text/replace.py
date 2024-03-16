import os

#遍历文件夹下的所有文件
def replace_text(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                text = file.read()
            #使用str的replace方法替换逗号和句号为无
            text = text.replace('，', '').replace('。', '').replace('、', '').replace('？', '').replace('！', '')
            with open(os.path.join(folder_path, filename), 'w') as file:
                file.write(text)

def main():
    # 指定文件夹路径
    folder_paths=['','','']
    for folder_path in folder_paths:
        replace_text(folder_path)
        print(f"{folder_path}替换完成")