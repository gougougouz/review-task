#!/bin/bash

while true; do
    echo "请选择操作："
    echo "1) 添加用户"
    echo "2) 删除用户"
    echo "3) 退出"
    read -p "请输入选项（1/2/3）: " option

    case $option in
        1)
            if [[ ! -f "account.txt" ]]; then
                echo "错误：account.txt不存在。"
                continue
            fi
            > output.txt  # 清空输出文件
            while IFS= read -r username; do
                username=$(echo "$username" | xargs)  # 去除前后空格
                [[ -z "$username" ]] && continue  # 跳过空行
                if id "$username" &>/dev/null; then
                    echo "用户 $username 已存在，跳过。"
                else
                    password=$(openssl rand -base64 6)
                    if ! useradd -m "$username"; then
                        echo "创建用户 $username 失败。"
                        continue
                    fi
                    echo "$password" | passwd --stdin "$username" &>/dev/null
                    chage -d 0 "$username"
                    echo "$username $password" >> output.txt
                    echo "用户 $username 添加成功，初始密码：$password"
                fi
            done < account.txt
            echo "用户添加完成。输出文件：output.txt"
            ;;
        2)
            if [[ ! -f "account.txt" ]]; then
                echo "错误：account.txt不存在。"
                continue
            fi
            while IFS= read -r username; do
                username=$(echo "$username" | xargs)
                [[ -z "$username" ]] && continue
                if id "$username" &>/dev/null; then
                    if userdel -r "$username"; then
                        echo "用户 $username 已删除。"
                    else
                        echo "删除用户 $username 失败。"
                    fi
                else
                    echo "用户 $username 不存在。"
                fi
            done < account.txt
            ;;
        3)
            echo "退出脚本。"
            exit 0
            ;;
        *)
            echo "无效选项，请重新输入。"
            ;;
    esac
done