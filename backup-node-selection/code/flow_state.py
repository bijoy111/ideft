"""
A real stateful network function: a connection-tracking / NAT flow table.

Why this exists
---------------
DEFT's shipped backup discards everything it is sent::

    @staticmethod
    def update_state(state):
        # with open('../data/slave_state.json', 'w') as f:
        #     json.dump(state, f)
        print('Received state from master: ')
        print(state)
        return True

The persistence is commented out, so the backup holds no state and could not take
over if the primary failed. The replication experiments in this paper consequently
ship a single hard-coded dictionary entry, ``DUMMY_STATE``, of about thirty bytes.
That is enough to measure the cost of an RPC round trip, which is what those
experiments set out to do, but it means the quantity the online analysis turns
on -- the cost of moving a primary's state to a different backup -- has no
empirical value at all. It is a free parameter.

This module supplies the missing piece: a flow table of the kind a stateful NAT or
connection tracker maintains, with the three operations replication actually needs.

  apply_delta   what ``update_state`` should do: fold an incremental update into
                the table, so the backup really is a warm standby.
  serialize     what a migration must transfer: the whole table, in a wire format.
  load          what the receiving backup must do before it can serve.

``serialize`` and ``load`` are what make the migration cost measurable, and
measuring it is the point of this module.

The 5-tuple key matches the format DEFT already uses for its dummy state
(``"10.0.0.1,10.0.0.2,1"`` is source, destination, protocol), extended with ports
so that entries are distinct per connection rather than per host pair.

Wire formats
------------
Two encodings are provided, because the choice changes the migration cost by a
large factor and DEFT's own choice is the expensive one.

  ``xmlrpc``  what DEFT actually puts on the wire: the XML-RPC marshalling of a
              dict of string keys, every integer wrapped in its own element.
              Note that XML-RPC's ``<int>`` is signed 32-bit, so it cannot carry
              an IPv4 address as an integer (anything from 128.0.0.0 up overflows)
              nor a byte counter past 2 GiB. Any real NAT state therefore has to
              be sent through this transport as strings, which is what the
              encoding below does and part of why it is as large as it is.
  ``json``    what DEFT's commented-out ``json.dump`` would have written, and a
              reasonable floor for a text format.
  ``packed``  fixed-width binary, 13 bytes of key and 31 bytes of value. This is
              what a system that cared about handoff time would send.

Reporting both separates "state transfer is expensive" from "this encoding of
state transfer is expensive", which are different claims with different fixes.
"""

import json
import struct
import xmlrpc.client
import time
from dataclasses import dataclass, field

# Protocol numbers, as they appear in the 5-tuple.
TCP, UDP, ICMP = 6, 17, 1

# struct formats. Key: src, dst (IPv4 as uint32), sport, dport, proto.
_KEY_FMT = "!IIHHB"          # 13 bytes
# Value: packets, bytes (uint64), nat_ip (uint32), nat_port (uint16),
# last_seen (double), tcp_state (uint8).
_VAL_FMT = "!QQIHdB"         # 31 bytes
KEY_BYTES = struct.calcsize(_KEY_FMT)
VAL_BYTES = struct.calcsize(_VAL_FMT)

# TCP connection states a tracker distinguishes. Only the count matters here --
# the field exists so an entry is the size a real entry is.
TCP_NEW, TCP_ESTABLISHED, TCP_FIN, TCP_CLOSED = 0, 1, 2, 3


@dataclass
class FlowEntry:
    """One connection's state, as a NAT or conntrack module would hold it."""
    packets: int = 0
    bytes: int = 0
    nat_ip: int = 0
    nat_port: int = 0
    last_seen: float = 0.0
    tcp_state: int = TCP_NEW

    def pack(self):
        return struct.pack(_VAL_FMT, self.packets, self.bytes, self.nat_ip,
                           self.nat_port, self.last_seen, self.tcp_state)

    @staticmethod
    def unpack(blob, offset=0):
        values = struct.unpack_from(_VAL_FMT, blob, offset)
        return FlowEntry(*values)


def _ipv4(value):
    """Dotted-quad text, the only way an IPv4 address fits through XML-RPC."""
    return "{}.{}.{}.{}".format((value >> 24) & 0xFF, (value >> 16) & 0xFF,
                                (value >> 8) & 0xFF, value & 0xFF)


def _un_ipv4(text):
    a, b, c, d = (int(x) for x in text.split("."))
    return (a << 24) | (b << 16) | (c << 8) | d


def key_to_str(key):
    """Render a 5-tuple in DEFT's comma-separated string form."""
    src, dst, sport, dport, proto = key
    return "{},{},{},{},{}".format(src, dst, sport, dport, proto)


def str_to_key(text):
    parts = text.split(",")
    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))


@dataclass
class FlowTable:
    """The network function's replicated state.

    ``table`` maps a 5-tuple to a FlowEntry. ``dirty`` tracks which keys have
    changed since the last delta was taken, which is what incremental replication
    needs to send and what DEFT's per-update ``update_state`` call stands in for.
    """
    table: dict = field(default_factory=dict)
    dirty: set = field(default_factory=set)
    # Counters, so experiments can report what the NF actually did rather than
    # what it was asked to do.
    applied_deltas: int = 0
    applied_entries: int = 0
    expired: int = 0

    def __len__(self):
        return len(self.table)

    # -- data-plane side -------------------------------------------------

    def observe(self, key, packet_bytes, now, tcp_state=None, nat=None):
        """Record a packet against a flow, creating the entry if it is new.

        This is the operation the primary performs; whatever it changes is what
        later has to be replicated.
        """
        entry = self.table.get(key)
        if entry is None:
            entry = FlowEntry(nat_ip=nat[0] if nat else 0,
                              nat_port=nat[1] if nat else 0)
            self.table[key] = entry
        entry.packets += 1
        entry.bytes += packet_bytes
        entry.last_seen = now
        if tcp_state is not None:
            entry.tcp_state = tcp_state
        self.dirty.add(key)
        return entry

    def expire(self, now, idle_timeout):
        """Drop flows idle for longer than the timeout, as a tracker must or the
        table grows without bound. Returns the number removed."""
        stale = [k for k, e in self.table.items() if now - e.last_seen > idle_timeout]
        for k in stale:
            del self.table[k]
            self.dirty.discard(k)
        self.expired += len(stale)
        return len(stale)

    # -- replication side ------------------------------------------------

    def take_delta(self):
        """Return the changed entries and clear the dirty set.

        This is the payload of one ``update_state`` call under incremental
        replication, and it is what the per-call latency experiments should be
        carrying instead of a fixed thirty-byte dictionary.
        """
        delta = {k: self.table[k] for k in self.dirty if k in self.table}
        self.dirty.clear()
        return delta

    def apply_delta(self, delta):
        """Fold a delta into this table. This is what the backup must do, and what
        DEFT's ``update_state`` stub does not do."""
        for key, entry in delta.items():
            self.table[key] = entry
        self.applied_deltas += 1
        self.applied_entries += len(delta)

    # -- handoff ---------------------------------------------------------

    def serialize(self, encoding="packed"):
        """The whole table, as bytes. Transferring this is the migration."""
        if encoding == "packed":
            parts = [struct.pack("!I", len(self.table))]
            for key, entry in self.table.items():
                parts.append(struct.pack(_KEY_FMT, *key))
                parts.append(entry.pack())
            return b"".join(parts)
        if encoding in ("json", "xmlrpc"):
            payload = {key_to_str(k): [e.packets, e.bytes, e.nat_ip, e.nat_port,
                                       e.last_seen, e.tcp_state]
                       for k, e in self.table.items()}
            if encoding == "json":
                return json.dumps(payload).encode("utf-8")
            # Exactly what Primary.replicate() hands to the transport: the
            # marshalled body of an update_state call. Fields that exceed
            # XML-RPC's signed 32-bit <int> are sent as strings, since the
            # protocol affords no other way to carry them.
            payload = {k: [str(v[0]), str(v[1]), _ipv4(v[2]), v[3], v[4], v[5]]
                       for k, v in payload.items()}
            return xmlrpc.client.dumps((payload,), "update_state").encode("utf-8")
        raise ValueError("unknown encoding: {}".format(encoding))

    @staticmethod
    def deserialize(blob, encoding="packed"):
        """Rebuild a table from bytes. The receiving backup must finish this before
        it can serve, so it is part of the migration cost, not an afterthought."""
        table = FlowTable()
        if encoding == "packed":
            (count,) = struct.unpack_from("!I", blob, 0)
            offset = 4
            step = KEY_BYTES + VAL_BYTES
            for _ in range(count):
                key = struct.unpack_from(_KEY_FMT, blob, offset)
                entry = FlowEntry.unpack(blob, offset + KEY_BYTES)
                table.table[key] = entry
                offset += step
            return table
        if encoding in ("json", "xmlrpc"):
            if encoding == "json":
                payload = json.loads(blob.decode("utf-8"))
                for text, values in payload.items():
                    table.table[str_to_key(text)] = FlowEntry(*values)
                return table
            (payload,), _ = xmlrpc.client.loads(blob.decode("utf-8"))
            for text, values in payload.items():
                table.table[str_to_key(text)] = FlowEntry(
                    int(values[0]), int(values[1]), _un_ipv4(values[2]),
                    values[3], values[4], values[5])
            return table
        raise ValueError("unknown encoding: {}".format(encoding))

    def bytes_per_entry(self, encoding="packed"):
        if not self.table:
            return 0.0
        return len(self.serialize(encoding)) / len(self.table)


def delta_to_wire(delta, encoding="packed"):
    """Encode a delta for transmission, in the same two formats."""
    scratch = FlowTable(table=dict(delta))
    return scratch.serialize(encoding)
