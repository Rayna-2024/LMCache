Cache identity and L1 xattrs
================================

Clients pass matching tags and xattrs through the existing
``IPCCacheServerKey.request_configs`` dictionary. The IPC key and its wire
schema retain their dev definitions.

.. code-block:: python

   request_configs = {
       "lmcache.tag.adapter_revision": "abc123",
       "lmcache.xattr.owner": "alice",
       "lmcache.xattr.priority": 1,
   }

The server extracts ``lmcache.tag.*`` into ``ObjectKey.identity.tags`` and
``lmcache.xattr.*`` into ``ObjectKey.xattrs``, removing each prefix. Unrelated
request configurations keep their existing meaning. Tags use nonempty names
and string values; xattrs preserve their request-config values.

``ObjectKeyIdentity`` contains ``chunk_hash``, ``kv_rank``, ``model_name``,
``object_group_id``, ``cache_salt`` and immutable, name-sorted ``tags``.
Consumers read matching fields through ``key.identity``, for example
``key.identity.model_name``.

``xattrs`` is mutable, copied per object and excluded from equality/hashing.
An L1 consumer can delete one attribute directly:

.. code-block:: python

   owner = key.xattrs.pop("owner", None)

Changing xattrs does not change a cache hit or create another entry.
Forwarding, store queues and temporary keys preserve xattrs; consumers decide
which entries to remove. P2P messages and operation traces carry xattrs.

The internal P2P schema assumes both services upgrade together and has no
legacy field reservations. Persistent key names, HTTP/control-plane identity
projections and coordinator captures/pins do not include xattrs.

L2 adapters use ``lmcache.v1.distributed.key_codec`` for shared storage names,
filenames and recovery. Untagged names remain unchanged; sorted tags use a
reversible token. Extended FS filenames must fit within 255 UTF-8 bytes.

Construct keys with ``ObjectKey(identity=identity)``, where ``identity`` is an
``ObjectKeyIdentity``. Update matching fields by replacing the nested identity:

.. code-block:: python

   from dataclasses import replace

   updated = replace(key, identity=replace(key.identity, object_group_id=1))

``EncodedObjectKey`` retains flat HTTP/JSON fields, with ``chunk_hash_hex``
instead of bytes and an optional ``tags`` map. Internal P2P services upgrade
together. IPC clients retain the dev message shape. This API does not activate
precision isolation in an engine integration.

Token-addressed control operations
==========================================

Coordinator lookup, pin/unpin, delete and prefetch requests accept a ``tags``
map whose entries match the ``lmcache.tag.*`` values used during storage.
Node prefetch and coordinator CacheBlend queries carry the same map. Empty
tags select untagged content and do not match every tag namespace.

.. code-block:: json

   {
     "model_name": "model",
     "world_size": 1,
     "token_ids": [1, 2, 3, 4],
     "cache_salt": "",
     "tags": {"adapter_revision": "abc123"}
   }

Tags require nonempty string names and string values. Control-plane responses
and delete keys include the same map alongside the existing flat fields.
