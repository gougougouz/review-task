#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

struct Character {
    int id;  // 角色的原始编号
    string name;  // 角色名称
    vector<int> attrs;  // 属性值数组
    long long power;  // 战力值
    
    Character(int _id, string _name, vector<int> _attrs, long long _power) 
        : id(_id), name(_name), attrs(_attrs), power(_power) {}
};

int main() {
    int n, m;
    cin >> n >> m;
    
    // 读取角色名称
    vector<string> names(n);
    for(int i = 0; i < n; i++) {
        cin >> names[i];
    }
    
    // 读取属性名称
    vector<string> attr_names(m);
    for(int i = 0; i < m; i++) {
        cin >> attr_names[i];
    }
    
    // 读取属性系数
    vector<int> coefficients(m);
    for(int i = 0; i < m; i++) {
        cin >> coefficients[i];
    }
    
    // 读取角色属性值并计算战力
    vector<Character> characters;
    for(int i = 0; i < n; i++) {
        vector<int> attrs(m);
        long long power = 0;
        for(int j = 0; j < m; j++) {
            cin >> attrs[j];
            power += 1LL * attrs[j] * coefficients[j];
        }
        characters.emplace_back(i + 1, names[i], attrs, power);
    }
    
    // 排序
    sort(characters.begin(), characters.end(), [&](const Character& a, const Character& b) {
        // 首先按战力值排序
        if(a.power != b.power) {
            return a.power > b.power;
        }
        
        // 战力值相同时，找到字典序最小的不同属性进行比较
        for(int j = 0; j < m; j++) {
            bool found = false;
            string min_attr_name;
            int attr_index = -1;
            
            // 找到字典序最小的不同属性
            for(int k = 0; k < m; k++) {
                if(a.attrs[k] != b.attrs[k]) {
                    if(!found || attr_names[k] < min_attr_name) {
                        found = true;
                        min_attr_name = attr_names[k];
                        attr_index = k;
                    }
                }
            }
            
            if(found) {
                return a.attrs[attr_index] > b.attrs[attr_index];
            }
        }
        
        // 所有属性值都相同时，按角色名称字典序排序
        return a.name < b.name;
    });
    
    // 输出排序后的编号
    for(int i = 0; i < n; i++) {
        cout << characters[i].id << (i == n-1 ? '\n' : ' ');
    }
    
    return 0;
}