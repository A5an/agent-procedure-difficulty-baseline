#!/bin/zsh
# usage: cr.sh doi...
for d in "$@"; do
  curl -s "https://api.crossref.org/works/$d" | python3 -c "
import sys,json
try:
  m=json.load(sys.stdin)['message']
  print('OK',m['DOI'],'|',(m.get('title') or [''])[0][:90],'|',(m.get('container-title') or [''])[0][:50],'|',(m.get('issued',{}).get('date-parts') or [[None]])[0][0],'|',', '.join(a.get('family','') for a in m.get('author',[])[:3]))
except Exception as e: print('FAIL','$d')
"
done
