from   pprint import pprint
import sys
import weightwatcher as ww

model = sys.argv[1]

watcher = ww.WeightWatcher(model=model)
details = watcher.analyze()
pprint(details)
summary = watcher.get_summary(details)
pprint(summary)
