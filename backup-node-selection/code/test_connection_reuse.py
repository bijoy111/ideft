"""Does one ServerProxy reuse one TCP connection across calls, as the paper states?
Count accept()s server-side while a single proxy makes 20 calls."""
import multiprocessing, socket, time, xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer

def serve(port, counter):
    class CountingServer(SimpleXMLRPCServer):
        def get_request(self):
            counter.value += 1
            return super().get_request()
    s = CountingServer(("127.0.0.1", port), logRequests=False)
    s.register_function(lambda state: True, "update_state")
    s.serve_forever()

if __name__ == "__main__":
    counter = multiprocessing.Value("i", 0)
    p = multiprocessing.Process(target=serve, args=(9701, counter), daemon=True)
    p.start(); time.sleep(0.6)
    proxy = xmlrpc.client.ServerProxy("http://127.0.0.1:9701")
    N = 20
    for _ in range(N):
        proxy.update_state({"k": 1})
    time.sleep(0.3)
    print("calls made            : {}".format(N))
    print("TCP accepts on server : {}".format(counter.value))
    print("=> {}".format("ONE persistent connection (paper correct)" if counter.value <= 2
                         else "a NEW TCP connection per call (paper's claim does not hold)"))
    p.terminate()
