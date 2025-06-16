#include <iostream>
using namespace std;

int countOnes(int n) {
    int count = 0;
    for (int i = 1; i <= n; i++) {
        int num = i;
        // 检查当前数字中每一位是否为 1
        while (num > 0) {
            if (num % 10 == 1) {
                count++;
            }
            num /= 10; // 去掉最低位
        }
    }
    return count;
}

int main() {
    int n;
    cin >> n; // 输入 n
    cout << countOnes(n) << endl; // 输出 1 出现的次数
    return 0;
}