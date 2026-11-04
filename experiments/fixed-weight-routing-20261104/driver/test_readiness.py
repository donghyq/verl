import asyncio,time,unittest
from types import SimpleNamespace as NS
from gateway_readiness import wait_workers

class ReadinessTests(unittest.TestCase):
    def test_empty_partial_and_unhealthy_are_waited_before_dispatch(self):
        rows=[[],[{'url':'a','id':'one','is_healthy':True}],
            [{'url':'a','id':'one','is_healthy':True},{'url':'b','id':'two','is_healthy':False}],
            [{'url':'a','id':'one','is_healthy':True},{'url':'b','id':'two','is_healthy':True}]]
        async def call(*a):return 200,{'workers':rows.pop(0)},{}
        actual=asyncio.run(wait_workers(NS(deadline=time.time()+5,call=call),'g',['a','b'],'test'))
        self.assertEqual(actual,{'a':'one','b':'two'});self.assertFalse(rows)
    def test_foreign_and_duplicate_workers_fail_closed(self):
        for workers in [[{'url':'unknown','id':'one','is_healthy':True}],
            [{'url':'a','id':'same','is_healthy':True},{'url':'b','id':'same','is_healthy':True}]]:
            async def call(*a):return 200,{'workers':workers},{}
            with self.assertRaises(ValueError):asyncio.run(wait_workers(NS(deadline=time.time()+5,call=call),'g',['a','b'],'bad'))
    def test_expired_absolute_deadline_never_reads_or_generates(self):
        async def call(*a):raise AssertionError('expired request was sent')
        with self.assertRaises(TimeoutError):asyncio.run(wait_workers(NS(deadline=time.time()-1,call=call),'g',['a','b'],'expired'))

if __name__=='__main__':unittest.main()
