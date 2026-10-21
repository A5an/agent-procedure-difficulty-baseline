#!/bin/zsh
q="$1"; curl -s -G "https://api.crossref.org/works" --data-urlencode "query.bibliographic=$q" --data-urlencode "rows=${2:-3}" --data-urlencode "select=DOI,title,container-title,issued,author" | python3 -c "
import sys,json
for m in json.load(sys.stdin)['message']['items']:
  print(' ',m['DOI'],'|',(m.get('title') or [''])[0][:80],'|',(m.get('container-title') or [''])[0][:40],'|',(m.get('issued',{}).get('date-parts') or [[None]])[0][0],'|',', '.join(a.get('family','') for a in m.get('author',[])[:2]))"
