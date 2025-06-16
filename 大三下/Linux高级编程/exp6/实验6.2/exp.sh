#!/bin/bash

# 创建测试文件和目录结构
mkdir -p test_dir
touch test_file.txt
touch test_dir/file1.txt
touch test_dir/file2.txt

echo "已创建测试文件和目录结构："
echo "  - test_file.txt"
echo "  - test_dir/"
echo "    - file1.txt"
echo "    - file2.txt"
echo ""

# 提示用户输入
read -p "请输入要处理的文件或目录名: " target

# 检查目标是否存在
if [ ! -e "$target" ]; then
    echo "错误: 文件或目录 '$target' 不存在!"
    exit 1
fi

# 处理文件
if [ -f "$target" ]; then
    # 获取文件扩展名
    extension="${target##*.}"
    
    case "$extension" in
        gz)
            # 解压.gz文件
            echo "正在解压 $target ..."
            gunzip "$target"
            echo "已解压并删除原压缩包"
            ;;
        bz2)
            # 解压.bz2文件
            echo "正在解压 $target ..."
            bunzip2 "$target"
            echo "已解压并删除原压缩包"
            ;;
        zip)
            # 解压.zip文件
            echo "正在解压 $target ..."
            unzip "$target"
            rm "$target"
            echo "已解压并删除原压缩包"
            ;;
        tar.gz|tgz)
            # 解压.tar.gz文件
            echo "正在解压 $target ..."
            tar -xzf "$target"
            rm "$target"
            echo "已解压并删除原压缩包"
            ;;
        *)
            # 其他扩展名，提供压缩选项
            echo "请选择压缩格式:"
            select comp_type in gz bz2 zip tar.gz; do
                case $comp_type in
                    gz)
                        echo "正在使用gzip压缩 $target ..."
                        gzip "$target"
                        echo "已压缩并删除原文件"
                        break
                        ;;
                    bz2)
                        echo "正在使用bzip2压缩 $target ..."
                        bzip2 "$target"
                        echo "已压缩并删除原文件"
                        break
                        ;;
                    zip)
                        echo "正在使用zip压缩 $target ..."
                        zip "${target%.*}.zip" "$target"
                        rm "$target"
                        echo "已压缩并删除原文件"
                        break
                        ;;
                    tar.gz)
                        echo "正在使用tar.gz压缩 $target ..."
                        tar -czf "${target%.*}.tar.gz" "$target"
                        rm "$target"
                        echo "已压缩并删除原文件"
                        break
                        ;;
                    *)
                        echo "无效选项，请重新选择"
                        ;;
                esac
            done
            ;;
    esac
# 处理目录
elif [ -d "$target" ]; then
    echo "请选择目录压缩格式:"
    select dir_comp_type in tar.gz zip; do
        case $dir_comp_type in
            tar.gz)
                echo "正在使用tar.gz压缩目录 $target ..."
                tar -czf "${target}.tar.gz" "$target"
                rm -r "$target"
                echo "已压缩并删除原目录"
                break
                ;;
            zip)
                echo "正在使用zip压缩目录 $target ..."
                zip -r "${target}.zip" "$target"
                rm -r "$target"
                echo "已压缩并删除原目录"
                break
                ;;
            *)
                echo "无效选项，请重新选择"
                ;;
        esac
    done
fi