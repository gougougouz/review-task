def max_watered_land(n, heights):
    max_watered = 0

    def flow(start):
        visited = set()
        stack = [start]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            if curr > 0 and heights[curr - 1] <= heights[curr]:
                stack.append(curr - 1)
            if curr < n - 1 and heights[curr + 1] <= heights[curr]:
                stack.append(curr + 1)
        return len(visited)

    for i in range(n):
        max_watered = max(max_watered, flow(i))

    return max_watered
def main():
    n = int(input())
    heights = list(map(int, input().split()))
    print(max_watered_land(n, heights))
    pass    

if __name__ == '__main__':
    main()