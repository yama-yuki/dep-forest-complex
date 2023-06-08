'''
find sentence len, avg number of nodes&edges in a forest created with 'N'bestEisner
'''

nlist = ['8','16','32','64','128']

for n in nlist:
    data = 'len'+n+'.out'

    with open(data, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sent_lens = 0
        edges = 0
        cnt = 0
        for line in lines:
            sent, edge = line.rstrip('\n').split(' ')
            if sent:
                #sent_lens.append(int(sent))
                #edges.append(int(edge))
                sent_lens+=int(sent)
                edges+=int(edge)
                cnt+=1
        

        print(data)
        #print('sent')
        print(sent_lens/cnt)
        #print('edge')
        print(edges/cnt)

        print(cnt)