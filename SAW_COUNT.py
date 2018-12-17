
total = 0
N = 20
visited = []

def go(x, y, i):
    global total
    visited
    
    if i == 0:
        total += 1
    else:
        neighbours = [[x+1, y],[x-1, y],[x, y+1],[x, y-1]]
        for neigh in neighbours:
            if neigh not in visited:
                visited.append(neigh)
                go(neigh[0], neigh[1], i-1)
    visited.remove([x, y])

startPos = [0, 0]
visited.append(startPos)
go(startPos[0], startPos[1], N)
print(total)