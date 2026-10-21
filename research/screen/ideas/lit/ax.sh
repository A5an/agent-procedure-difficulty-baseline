#!/bin/zsh
# usage: ax.sh "search query" [n]
curl -sL -G "https://export.arxiv.org/api/query" --data-urlencode "search_query=$1" --data-urlencode "max_results=${2:-6}" --data-urlencode "sortBy=relevance" | python3 -c "
import sys,re
t=sys.stdin.read()
for e in t.split('<entry>')[1:]:
  g=lambda k:re.sub(r'\s+',' ',re.search('<%s[^>]*>(.*?)</%s>'%(k,k),e,re.S).group(1))
  print(' ',g('id').split('abs/')[-1],g('published')[:7],'|',g('title')[:105])
"
sleep 5
