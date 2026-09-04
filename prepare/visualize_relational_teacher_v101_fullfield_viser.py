#!/usr/bin/env python3
"""Read-only Teacher-v10.1 viewer using the verified v10 panel renderer."""

from __future__ import annotations

import visualize_relational_teacher_v10_supervised_capacity_viser as viewer
from relational_teacher_v101_fullfield_contract import SCHEMA


def main() -> None:
    viewer.SCHEMA = SCHEMA
    viewer.main()


if __name__ == "__main__":
    main()
