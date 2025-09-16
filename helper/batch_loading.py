from weave.trace.vals import TableRef, WeaveTable, WeaveObject
from weave.trace_server.trace_server_interface import TableQueryReq, TableQueryStatsReq
from weave.trace.serialize import from_json
from weave.trace.vals import make_trace_obj, logger
from weave.trace.context.tests_context import get_raise_on_captured_errors
from tqdm import tqdm
import math
from collections.abc import Generator

def load_dataset(table:WeaveTable|WeaveObject, page_size=1000):
    table = table
    if isinstance(table, WeaveObject):
        table = table.rows

    page_index = 0

    rows = list()
    rows_digest = list()

    stat_resp = table.server.table_query_stats(
                TableQueryStatsReq(
                    project_id=table.table_ref.project_id,
                    digest=table.table_ref.digest
                )
            )

    for batch_index in tqdm(range(math.ceil(stat_resp.count/page_size))):
        response = table.server.table_query(
                TableQueryReq(
                    project_id=table.table_ref.project_id,
                    digest=table.table_ref.digest,
                    offset=batch_index * page_size,
                    limit=page_size,
                    # filter=self.filter,
                )
            )
        
        for i, item in enumerate(response.rows):
            new_ref = table.ref.with_item(item.digest) if table.ref else None
            # Here, we use the raw rows if they exist, otherwise we use the
            # rows from the server. This is a temporary trick to ensure
            # we don't re-deserialize the rows on every access. Once all servers
            # return digests, this branch can be removed because anytime we have prefetched
            # rows we should also have the digests - and we should be in the
            #  _local_iter_with_remote_fallback case.
            val = (
                item.val
                if table._prefetched_rows is None
                else table._prefetched_rows[i]
            )
            res = from_json(val, table.table_ref.project_id, table.server)
            res = make_trace_obj(res, new_ref, table.server, table.root)
            rows.append(res)
            rows_digest.append(item.digest)
        
        if len(response.rows) < page_size:
            print("Last page:", batch_index,"/", math.ceil(stat_resp.count/page_size))
            break

    return rows, rows_digest

def custom_remote_iter(page_size=1000):
    def _remote_iter(cls) -> Generator[dict, None, None]:
        if cls.table_ref is None:
            return

        stat_resp = cls.server.table_query_stats(
                TableQueryStatsReq(
                    project_id=cls.table_ref.project_id,
                    digest=cls.table_ref.digest
                )
            )

        for page_index in tqdm(range(math.ceil(stat_resp.count/page_size))):
            response = cls.server.table_query(
                TableQueryReq(
                    project_id=cls.table_ref.project_id,
                    digest=cls.table_ref.digest,
                    offset=page_index * page_size,
                    limit=page_size,
                    # filter=self.filter,
                )
            )

            if cls._prefetched_rows is not None and len(response.rows) != len(
                cls._prefetched_rows
            ):
                if get_raise_on_captured_errors():
                    raise
                logger.error(
                    f"Expected length of response rows ({len(response.rows)}) to match prefetched rows ({len(cls._prefetched_rows)}). Ignoring prefetched rows."
                )
                cls._prefetched_rows = None

            for i, item in enumerate(response.rows):
                new_ref = cls.ref.with_item(item.digest) if cls.ref else None
                # Here, we use the raw rows if they exist, otherwise we use the
                # rows from the server. This is a temporary trick to ensure
                # we don't re-deserialize the rows on every access. Once all servers
                # return digests, this branch can be removed because anytime we have prefetched
                # rows we should also have the digests - and we should be in the
                #  _local_iter_with_remote_fallback case.
                val = (
                    item.val
                    if cls._prefetched_rows is None
                    else cls._prefetched_rows[i]
                )
                res = from_json(val, cls.table_ref.project_id, cls.server)
                res = make_trace_obj(res, new_ref, cls.server, cls.root)
                yield res

            # if len(response.rows) < page_size:
            #     break

            page_index += 1
    return _remote_iter