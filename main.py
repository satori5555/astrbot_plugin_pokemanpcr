from __future__ import annotations

import io
import math
import os
import random
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from . import _pcr_data


BLACKLIST_CARD = {
    'icon_unit_100031.png',
    'icon_unit_403111.png',
    'icon_unit_191711.png',
    'icon_unit_110231.png',
    'icon_unit_107311.png',
    'icon_unit_107331.png',
}
STAR_TO_RARITY = {'1': -1, '3': 0, '6': 1}
RARITY_DESC_TO_RARITY = {'普通': -1, '稀有': 0, '超稀有': 1}
RARITY_LABEL = {1: '超稀有', 0: '稀有', -1: '普通'}
RARITY_FRAME = {1: 'superrare.png', 0: 'rare.png', -1: 'normal.png'}
MIX_PROBABILITY = {
    str(list((-1, -1))): [0.8, 0.194, 0.006],
    str(list((-1, 0))): [0.44, 0.5, 0.06],
    str(list((-1, 1))): [0.55, 0.3, 0.1],
    str(list((0, 0))): [0.1, 0.8, 0.1],
    str(list((0, 1))): [0.3, 0.5, 0.2],
    str(list((1, 1))): [0.15, 0.25, 0.6],
}
OK_MIX_PROBABILITY = {
    str(list((-1, -1))): [0.846, 0.15, 0.004],
    str(list((-1, 0))): [0.56, 0.4, 0.04],
    str(list((-1, 1))): [0.68, 0.24, 0.08],
    str(list((0, 0))): [0.33, 0.6, 0.07],
    str(list((0, 1))): [0.44, 0.4, 0.16],
    str(list((1, 1))): [0.2, 0.3, 0.5],
}

POKE_ALIASES = {'戳', 'poke'}
MIX_ALIASES = {'献祭', '合成', '融合', 'mix'}
AUTO_MIX_ALIASES = {'一键献祭', '一键合成', '一键融合', '全部献祭', '全部合成', '全部融合', 'automix'}
EXCHANGE_ALIASES = {'交换', '交易', '互换', 'exchange'}
CONFIRM_EXCHANGE_ALIASES = {'确认交换', '同意交换', 'confirm_exchange'}
GIVE_ALIASES = {'赠送', '白给', '白送', 'give'}
COMPARE_ALIASES = {'仓库对比', 'compare'}
STORAGE_ALIASES = {'查看仓库', 'storage'}
REFRESH_ALIASES = {'刷新卡片', 'refresh_cards'}
ALL_COMMAND_ALIASES = (
    POKE_ALIASES
    | MIX_ALIASES
    | AUTO_MIX_ALIASES
    | EXCHANGE_ALIASES
    | CONFIRM_EXCHANGE_ALIASES
    | GIVE_ALIASES
    | COMPARE_ALIASES
    | STORAGE_ALIASES
    | REFRESH_ALIASES
)


@dataclass
class Outgoing:
    text: str = ''
    image_bytes: bytes | None = None
    at_ids: list[str] = field(default_factory=list)


@dataclass
class CardEntry:
    card_id: int
    chara_id: int
    rarity: int
    asset_path: str | None = None


@dataclass
class ExchangeRequest:
    sender_uid: str
    card1_id: int
    card1_name: str
    target_uid: str
    card2_id: int
    card2_name: str
    request_time: datetime = field(default_factory=datetime.now)


class CardRecordDAO:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self.connect() as conn:
            conn.execute(
                'CREATE TABLE IF NOT EXISTS card_record('
                'scope TEXT NOT NULL, uid TEXT NOT NULL, cid INTEGER NOT NULL, num INTEGER NOT NULL, '
                'PRIMARY KEY(scope, uid, cid))'
            )
            conn.execute(
                'CREATE TABLE IF NOT EXISTS limiter('
                'key TEXT NOT NULL PRIMARY KEY, num INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT "")'
            )

    def get_card_num(self, scope: str, uid: str, cid: int) -> int:
        with self.connect() as conn:
            row = conn.execute(
                'SELECT num FROM card_record WHERE scope=? AND uid=? AND cid=?',
                (scope, uid, cid),
            ).fetchone()
            return int(row[0]) if row else 0

    def add_card_num(self, scope: str, uid: str, cid: int, increment: int = 1) -> int:
        num = self.get_card_num(scope, uid, cid) + increment
        if num < 0:
            num = 0
        with self.connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO card_record (scope, uid, cid, num) VALUES (?, ?, ?, ?)',
                (scope, uid, cid, num),
            )
        return num

    def get_cards_num(self, scope: str, uid: str) -> dict[int, int]:
        with self.connect() as conn:
            rows = conn.execute(
                'SELECT cid, num FROM card_record WHERE scope=? AND uid=? AND num>0',
                (scope, uid),
            ).fetchall()
        return {int(cid): int(num) for cid, num in rows} if rows else {}

    def get_surplus_cards(self, scope: str, uid: str) -> dict[int, int]:
        with self.connect() as conn:
            rows = conn.execute(
                'SELECT cid, num FROM card_record WHERE scope=? AND uid=? AND num>1',
                (scope, uid),
            ).fetchall()
        return {int(cid): int(num) - 1 for cid, num in rows} if rows else {}

    def get_scope_ranking(self, scope: str, uid: str) -> int:
        with self.connect() as conn:
            rows = conn.execute(
                'SELECT uid FROM card_record WHERE scope=? AND num>0',
                (scope,),
            ).fetchall()
        if not rows:
            return -1
        counts = Counter([str(row[0]) for row in rows])
        if uid not in counts:
            return -1
        my_count = counts[uid]
        return sum(v > my_count for v in counts.values()) + 1

    def _ensure_key(self, key: str):
        with self.connect() as conn:
            conn.execute(
                'INSERT OR IGNORE INTO limiter (key, num, date) VALUES (?, 0, "")',
                (key,),
            )

    def get_num(self, key: str) -> int:
        self._ensure_key(key)
        with self.connect() as conn:
            row = conn.execute('SELECT num FROM limiter WHERE key=?', (key,)).fetchone()
        return int(row[0]) if row else 0

    def clear_key(self, key: str):
        self._ensure_key(key)
        with self.connect() as conn:
            conn.execute('UPDATE limiter SET num=0 WHERE key=?', (key,))

    def increment_key(self, key: str, num: int):
        self._ensure_key(key)
        with self.connect() as conn:
            conn.execute('UPDATE limiter SET num=num+? WHERE key=?', (num, key))

    def get_date(self, key: str) -> str:
        self._ensure_key(key)
        with self.connect() as conn:
            row = conn.execute('SELECT date FROM limiter WHERE key=?', (key,)).fetchone()
        return str(row[0]) if row else ''

    def set_date(self, date_value: str, key: str):
        self._ensure_key(key)
        with self.connect() as conn:
            conn.execute('UPDATE limiter SET date=? WHERE key=?', (date_value, key))


class SimpleCooldownLimiter:
    def __init__(self, cooldown_seconds: int):
        self.cooldown_seconds = int(cooldown_seconds)
        self._next_available: dict[str, float] = {}

    def check(self, key: str) -> bool:
        if self.cooldown_seconds <= 0:
            return True
        return time.time() >= self._next_available.get(key, 0.0)

    def start_cd(self, key: str):
        if self.cooldown_seconds <= 0:
            return
        self._next_available[key] = time.time() + self.cooldown_seconds


class DailyAmountLimiter:
    def __init__(self, dao: CardRecordDAO, limiter_type: str, max_num: int, reset_hour: int):
        self.dao = dao
        self.limiter_type = limiter_type
        self.max = int(max_num)
        self.reset_hour = int(reset_hour)

    def _full_key(self, key: tuple[str, str]) -> str:
        return f'{key[0]}::{key[1]}::{self.limiter_type}'

    def _day_key(self) -> str:
        now = datetime.now()
        shifted = now - timedelta(hours=self.reset_hour)
        return shifted.strftime('%Y-%m-%d')

    def _normalize(self, key: tuple[str, str]):
        full_key = self._full_key(key)
        day_key = self._day_key()
        if self.dao.get_date(full_key) != day_key:
            self.dao.set_date(day_key, full_key)
            self.dao.clear_key(full_key)

    def check(self, key: tuple[str, str]) -> bool:
        self._normalize(key)
        return self.dao.get_num(self._full_key(key)) < self.max

    def get_num(self, key: tuple[str, str]) -> int:
        self._normalize(key)
        return self.dao.get_num(self._full_key(key))

    def increase(self, key: tuple[str, str], num: int = 1):
        self._normalize(key)
        self.dao.increment_key(self._full_key(key), int(num))

    def reset(self, key: tuple[str, str]):
        self._normalize(key)
        self.dao.clear_key(self._full_key(key))


class ExchangeRequestMaster:
    def __init__(self, max_valid_time: int):
        self.last_exchange_request: dict[tuple[str, str], ExchangeRequest] = {}
        self.max_valid_time = int(max_valid_time)

    def add_exchange_request(self, scope: str, uid: str, request: ExchangeRequest):
        self.last_exchange_request[(scope, uid)] = request

    def has_exchange_request_to_confirm(self, scope: str, uid: str) -> bool:
        req = self.last_exchange_request.get((scope, uid))
        if not req:
            return False
        return (datetime.now() - req.request_time).total_seconds() <= self.max_valid_time

    def get_exchange_request(self, scope: str, uid: str) -> ExchangeRequest | None:
        if self.has_exchange_request_to_confirm(scope, uid):
            return self.last_exchange_request.get((scope, uid))
        self.delete_exchange_request(scope, uid)
        return None

    def delete_exchange_request(self, scope: str, uid: str):
        self.last_exchange_request.pop((scope, uid), None)


class PcrCharaResolver:
    def __init__(self):
        self.alias_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {}
        self._build_indexes()

    def _build_indexes(self):
        for chara_id, names in _pcr_data.CHARA_NAME.items():
            if chara_id == 1000 or chara_id in getattr(_pcr_data, 'UnavailableChara', set()):
                continue
            if not names:
                continue
            canonical = str(names[0]).strip()
            self.id_to_name[int(chara_id)] = canonical
            for alias in names:
                alias = str(alias).strip()
                if alias:
                    self.alias_to_id.setdefault(alias.lower(), int(chara_id))

    def name2id(self, name: str) -> int:
        return self.alias_to_id.get(name.strip().lower(), 0)

    def fromid(self, chara_id: int) -> str:
        return self.id_to_name.get(int(chara_id), f'未知角色{chara_id}')

    def all_ids(self) -> list[int]:
        return sorted(self.id_to_name.keys())


class CardCatalog:
    def __init__(
        self,
        resolver: PcrCharaResolver,
        unit_image_dir: str,
        preload_images: bool,
        use_placeholder_when_missing: bool,
    ):
        self.resolver = resolver
        self.unit_image_dir = Path(unit_image_dir).expanduser() if unit_image_dir else None
        self.preload_images = bool(preload_images)
        self.use_placeholder_when_missing = bool(use_placeholder_when_missing)
        self.entries_by_rarity: dict[int, list[CardEntry]] = {-1: [], 0: [], 1: []}
        self.entries_by_card_id: dict[int, CardEntry] = {}
        self.image_cache: dict[str, Image.Image] = {}
        self.total_card_ids: list[int] = []
        self.refresh()

    @staticmethod
    def build_card_id(rarity: int, chara_id: int) -> int:
        return 30000 + (int(rarity) * 1000) + int(chara_id)

    @staticmethod
    def get_card_rarity(card_id: int) -> int:
        if 33000 > card_id > 32000:
            return 1
        if card_id < 31000:
            return -1
        return 0

    def refresh(self) -> dict[str, int]:
        self.entries_by_rarity = {-1: [], 0: [], 1: []}
        self.entries_by_card_id = {}
        self.total_card_ids = []
        self.image_cache.clear()
        real_asset_count = 0

        if self.unit_image_dir and self.unit_image_dir.exists() and self.unit_image_dir.is_dir():
            for image_path in sorted(self.unit_image_dir.iterdir()):
                name = image_path.name
                if not name.startswith('icon_unit_') or name in BLACKLIST_CARD:
                    continue
                if len(name) < 16:
                    continue
                try:
                    chara_id = int(name[10:14])
                except ValueError:
                    continue
                if chara_id == 1000 or chara_id not in self.resolver.id_to_name:
                    continue
                star = name[14]
                if star not in STAR_TO_RARITY or name[15] != '1':
                    continue
                rarity = STAR_TO_RARITY[star]
                card_id = self.build_card_id(rarity, chara_id)
                entry = CardEntry(card_id=card_id, chara_id=chara_id, rarity=rarity, asset_path=str(image_path))
                self.entries_by_rarity[rarity].append(entry)
                self.entries_by_card_id[card_id] = entry
                self.total_card_ids.append(card_id)
                real_asset_count += 1
                if self.preload_images:
                    try:
                        img = Image.open(image_path)
                        self.image_cache[str(image_path)] = img.convert('RGBA') if img.mode != 'RGBA' else img.copy()
                    except Exception as exc:
                        logger.warning(f'预加载头像失败: {image_path} -> {exc}')

        if not self.entries_by_card_id and self.use_placeholder_when_missing:
            for chara_id in self.resolver.all_ids():
                for rarity in (-1, 0, 1):
                    card_id = self.build_card_id(rarity, chara_id)
                    entry = CardEntry(card_id=card_id, chara_id=chara_id, rarity=rarity, asset_path=None)
                    self.entries_by_rarity[rarity].append(entry)
                    self.entries_by_card_id[card_id] = entry
                    self.total_card_ids.append(card_id)

        return {
            'normal': len(self.entries_by_rarity[-1]),
            'rare': len(self.entries_by_rarity[0]),
            'super_rare': len(self.entries_by_rarity[1]),
            'real_assets': real_asset_count,
        }

    def has_card(self, rarity: int, chara_id: int) -> bool:
        return self.build_card_id(rarity, chara_id) in self.entries_by_card_id

    def resolve_card_id_by_name(self, card_name: str) -> int:
        card_name = card_name.strip()
        if not card_name:
            return 0
        if card_name.startswith('超稀有'):
            rarity = 1
            raw_name = card_name[3:]
        elif card_name.startswith('稀有'):
            rarity = 0
            raw_name = card_name[2:]
        else:
            rarity = -1
            raw_name = card_name[2:] if card_name.startswith('普通') else card_name
        chara_id = self.resolver.name2id(raw_name)
        if not chara_id:
            return 0
        card_id = self.build_card_id(rarity, chara_id)
        return card_id if card_id in self.entries_by_card_id else 0

    def get_chara_name(self, card_id: int) -> tuple[str, str, str]:
        rarity = self.get_card_rarity(card_id)
        chara_id = card_id % 1000 + (card_id // 1000 % 10) * 1000
        # 由 card_id 反推时，原公式可直接取 card_id % 10000 再按区间修正；这里直接用 entry 更稳。
        entry = self.entries_by_card_id.get(card_id)
        if entry:
            chara_id = entry.chara_id
        name = self.resolver.fromid(chara_id)
        desc = {1: '【超稀有】的', 0: '【稀有】的', -1: '【普通】的'}[rarity]
        return desc, name, f'{RARITY_LABEL[rarity]}{name}'

    def get_card_name_with_rarity(self, card_name: str) -> str:
        card_id = self.resolve_card_id_by_name(card_name)
        if card_id:
            return self.get_chara_name(card_id)[2]
        card_name = card_name.strip()
        if card_name.startswith('超稀有') or card_name.startswith('稀有') or card_name.startswith('普通'):
            return card_name
        return f'普通{card_name}'

    def random_entry(self, super_rare_prob: float, rare_prob: float) -> CardEntry:
        r = random.random()
        if r < super_rare_prob:
            pool = self.entries_by_rarity[1]
        elif r < super_rare_prob + rare_prob:
            pool = self.entries_by_rarity[0]
        else:
            pool = self.entries_by_rarity[-1]
        if not pool:
            for fallback in (1, 0, -1):
                if self.entries_by_rarity[fallback]:
                    pool = self.entries_by_rarity[fallback]
                    break
        return random.choice(pool)

    def entries_of_rarity(self, rarity: int) -> list[CardEntry]:
        return list(self.entries_by_rarity.get(int(rarity), []))

    def get_entry(self, card_id: int) -> CardEntry | None:
        return self.entries_by_card_id.get(int(card_id))


class PokeManPCRPlugin(Star):
    DEFAULTS = {
        'unit_image_dir': '',
        'preload_images': True,
        'use_placeholder_when_missing': True,
        'poke_get_cards_probability': 0.75,
        'poke_daily_limit': 3,
        'rare_probability': 0.17,
        'super_rare_probability': 0.03,
        'request_valid_time': 60,
        'poke_tip_limit': 1,
        'tip_cd_limit': 10 * 60,
        'poke_cooling_time': 3,
        'give_daily_limit': 3,
        'reset_hour': 0,
        'col_num': 17,
        'omit_threshold': 20,
    }

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self.plugin_name = 'astrbot_plugin_pokemanpcr'
        self.plugin_root = Path(__file__).resolve().parent
        self.assets_dir = self.plugin_root / 'assets'
        self.data_dir = Path(get_astrbot_plugin_data_path()) / self.plugin_name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = CardRecordDAO(self.data_dir / 'poke_man_pcr.db')
        self.resolver = PcrCharaResolver()
        self.catalog = CardCatalog(
            resolver=self.resolver,
            unit_image_dir=self._resolve_unit_image_dir(),
            preload_images=self._cfg('preload_images', bool),
            use_placeholder_when_missing=self._cfg('use_placeholder_when_missing', bool),
        )
        self.exchange_request_master = ExchangeRequestMaster(self._cfg('request_valid_time', int))
        self.poke_tip_cd_limiter = SimpleCooldownLimiter(self._cfg('tip_cd_limit', int))
        self.cooling_time_limiter = SimpleCooldownLimiter(self._cfg('poke_cooling_time', int))
        self.daily_tip_limiter = DailyAmountLimiter(self.db, 'tip', self._cfg('poke_tip_limit', int), self._cfg('reset_hour', int))
        self.daily_limiter = DailyAmountLimiter(self.db, 'poke', self._cfg('poke_daily_limit', int), self._cfg('reset_hour', int))
        self.daily_give_limiter = DailyAmountLimiter(self.db, 'give', self._cfg('give_daily_limit', int), self._cfg('reset_hour', int))
        self.font = self._load_font(16)
        self.placeholder_font = self._load_font(14)

    def _resolve_unit_image_dir(self) -> str:
        configured = self._cfg('unit_image_dir', str).strip()
        candidates: list[Path] = []

        if configured:
            configured_path = Path(configured).expanduser()
            if not configured_path.is_absolute():
                configured_path = (self.plugin_root / configured_path).resolve()
            candidates.append(configured_path)

        candidates.extend([
            self.plugin_root / 'unit__',
            self.plugin_root / 'unit',
        ])

        seen: set[str] = set()
        for candidate in candidates:
            candidate_key = str(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            try:
                if candidate.exists() and candidate.is_dir():
                    logger.info(f'[{self.plugin_name}] 使用头像目录: {candidate}')
                    return str(candidate)
            except Exception as exc:
                logger.warning(f'[{self.plugin_name}] 检查头像目录失败: {candidate} -> {exc}')

        if configured:
            logger.warning(f'[{self.plugin_name}] 配置的头像目录不存在，将回退到占位卡面: {configured}')
        else:
            logger.info(f'[{self.plugin_name}] 未找到同级 unit__/unit 头像目录，将使用占位卡面。')
        return ''

    def _cfg(self, key: str, caster):
        value = self.config.get(key, self.DEFAULTS[key]) if isinstance(self.config, dict) else self.DEFAULTS[key]
        if caster is bool:
            return bool(value)
        if caster is int:
            try:
                return int(value)
            except Exception:
                return int(self.DEFAULTS[key])
        if caster is float:
            try:
                return float(value)
            except Exception:
                return float(self.DEFAULTS[key])
        if caster is str:
            return '' if value is None else str(value)
        return value

    @staticmethod
    def _load_font(size: int):
        candidates = [
            'arial.ttf',
            'msyh.ttc',
            'simhei.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
        return ImageFont.load_default()

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        stats = self.catalog.refresh()
        logger.info(
            f'[{self.plugin_name}] 卡池已加载: 普通={stats["normal"]}, 稀有={stats["rare"]}, 超稀有={stats["super_rare"]}, 真实头像={stats["real_assets"]}'
        )

    @filter.command('pcrcards', alias={'pcr卡片'})
    async def pcr_help(self, event: AstrMessageEvent):
        yield event.plain_result(self._help_text())

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        if self._is_self_message(event):
            return
        outgoing = await self._dispatch(event)
        if not outgoing:
            return
        event.stop_event()
        if isinstance(outgoing, list):
            for item in outgoing:
                yield self._to_result(event, item)
            return
        yield self._to_result(event, outgoing)

    def _help_text(self) -> str:
        return (
            'PCR卡片插件已加载。\n'
            '支持：\n'
            '1. 戳\n'
            '2. 查看仓库 [@某人]\n'
            '3. 仓库对比 @某人\n'
            '4. 合成 卡1 卡2\n'
            '5. 一键合成 稀有度1 稀有度2 [轮数]\n'
            '6. 赠送 @某人 卡名\n'
            '7. 交换 卡1 @某人 卡2\n'
            '8. 确认交换\n'
            '9. 刷新卡片\n\n'
            '兼容原项目的纯中文消息，也兼容前面加 / 的写法。'
        )

    @staticmethod
    def _is_self_message(event: AstrMessageEvent) -> bool:
        try:
            msg = event.message_obj
            return str(getattr(msg, 'self_id', '')) == str(event.get_sender_id())
        except Exception:
            return False

    @staticmethod
    def _segment_type(seg) -> str:
        return str(getattr(seg, 'type', '') or '')

    def _is_at_segment(self, seg) -> bool:
        t = self._segment_type(seg).lower()
        return 'at' in t

    def _is_poke_segment(self, seg) -> bool:
        t = self._segment_type(seg)
        return 'Poke' in t or 'poke' in t

    def _segments(self, event: AstrMessageEvent) -> list:
        try:
            return list(getattr(event.message_obj, 'message', []) or [])
        except Exception:
            return []

    def _normalized_content(self, event: AstrMessageEvent) -> str:
        content = (getattr(event, 'message_str', '') or '').strip()
        return content[1:].strip() if content.startswith('/') else content

    def _scope_id(self, event: AstrMessageEvent) -> str:
        msg = event.message_obj
        scope = getattr(msg, 'group_id', '') or getattr(msg, 'session_id', '') or str(event.get_sender_id())
        return str(scope)

    def _sender_id(self, event: AstrMessageEvent) -> str:
        return str(event.get_sender_id())

    def _extract_at_ids(self, event: AstrMessageEvent) -> list[str]:
        ids: list[str] = []
        for seg in self._segments(event):
            if self._is_at_segment(seg):
                qq = getattr(seg, 'qq', None)
                if qq is not None:
                    ids.append(str(qq))
        return ids

    def _split_command(self, event: AstrMessageEvent) -> tuple[str, str]:
        content = self._normalized_content(event)
        if not content:
            return '', ''
        parts = content.split(maxsplit=1)
        cmd = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ''
        return cmd, rest

    def _strip_alias_prefix(self, text: str, aliases: Iterable[str]) -> str:
        text = text.strip()
        if text.startswith('/'):
            text = text[1:].strip()
        for alias in sorted(set(aliases), key=len, reverse=True):
            if text == alias:
                return ''
            if text.startswith(alias + ' '):
                return text[len(alias):].strip()
        return text

    def _to_result(self, event: AstrMessageEvent, outgoing: Outgoing):
        if outgoing.image_bytes is None and not outgoing.at_ids:
            return event.plain_result(outgoing.text)
        chain = []
        for uid in outgoing.at_ids:
            chain.append(Comp.At(qq=uid))
        if outgoing.at_ids and outgoing.text:
            chain.append(Comp.Plain(' '))
        if outgoing.text:
            chain.append(Comp.Plain(outgoing.text))
        if outgoing.image_bytes is not None:
            chain.append(Comp.Image.fromBytes(outgoing.image_bytes))
        return event.chain_result(chain)

    async def _dispatch(self, event: AstrMessageEvent) -> Outgoing | list[Outgoing] | None:
        segments = self._segments(event)
        for seg in segments:
            if self._is_poke_segment(seg):
                return self._handle_poke(event)

        cmd, rest = self._split_command(event)
        if not cmd and not segments:
            return None

        if cmd in POKE_ALIASES:
            return self._handle_poke(event)
        if cmd in STORAGE_ALIASES:
            return self._handle_storage(event)
        if cmd in COMPARE_ALIASES:
            return self._handle_storage_compare(event)
        if cmd in MIX_ALIASES:
            return self._handle_mix(event, rest)
        if cmd in AUTO_MIX_ALIASES:
            return self._handle_auto_mix(event, rest)
        if cmd in EXCHANGE_ALIASES:
            return self._handle_exchange(event)
        if cmd in CONFIRM_EXCHANGE_ALIASES:
            return self._handle_confirm_exchange(event)
        if cmd in GIVE_ALIASES:
            return self._handle_give(event)
        if cmd in REFRESH_ALIASES:
            return self._handle_refresh_cards()

        # 兼容原插件：纯文字 “戳”
        content = self._normalized_content(event)
        if content == '戳':
            return self._handle_poke(event)
        return None

    @staticmethod
    def _roll_cards_amount() -> int:
        roll = random.random()
        if roll <= 0.01:
            return 10
        if roll <= 0.1:
            return 5
        if roll <= 0.3:
            return 4
        if roll <= 0.7:
            return 3
        if roll <= 0.9:
            return 2
        return 1

    @staticmethod
    def _roll_extra_bonus() -> int:
        roll = random.random()
        if roll < 0.01:
            return 3
        if roll < 0.1:
            return 2
        return 1

    @staticmethod
    def _normalize_digit_format(n: int) -> str:
        return f'0{n}' if n < 10 else str(n)

    def _load_unit_image(self, entry: CardEntry) -> Image.Image:
        if entry.asset_path:
            cached = self.catalog.image_cache.get(entry.asset_path)
            if cached is not None:
                return cached.copy()
            img = Image.open(entry.asset_path)
            img = img.convert('RGBA') if img.mode != 'RGBA' else img
            return img
        return self._build_placeholder_unit_image(entry)

    def _build_placeholder_unit_image(self, entry: CardEntry) -> Image.Image:
        base_color = {
            -1: (245, 245, 245, 255),
            0: (220, 235, 255, 255),
            1: (255, 235, 245, 255),
        }[entry.rarity]
        img = Image.new('RGBA', (80, 80), base_color)
        draw = ImageDraw.Draw(img)
        name = self.resolver.fromid(entry.chara_id)
        text = name[:4]
        bbox = draw.textbbox((0, 0), text, font=self.placeholder_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((80 - tw) / 2, (80 - th) / 2), text, fill=(40, 40, 40), font=self.placeholder_font)
        return img

    def _frame_image(self, frame_name: str, size: tuple[int, int] = (80, 80)) -> Image.Image:
        path = self.assets_dir / frame_name
        frame = Image.open(path)
        return frame.resize(size, Image.Resampling.LANCZOS)

    def _add_rarity_frame(self, img: Image.Image, rarity: int) -> Image.Image:
        img = img.copy().resize((80, 80), Image.Resampling.LANCZOS)
        frame = self._frame_image(RARITY_FRAME[rarity])
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')
        img.paste(frame, (0, 0), mask=frame.split()[3])
        return img

    def _draw_num_text(self, img: Image.Image, num: int, draw_base_color: bool, color: tuple[int, int, int], offset_x: int, offset_y: int) -> Image.Image:
        img = img.copy()
        draw = ImageDraw.Draw(img)
        text = f'×{num}'
        if len(text) == 2:
            offset_r = 0
            offset_t = 0
        else:
            offset_r = 10
            offset_t = 9
        if draw_base_color:
            draw.rectangle((59 - offset_r, 60, 75, 77), fill=(255, 255, 255))
            draw.rectangle((59 - offset_r, 60, 77, 75), fill=(255, 255, 255))
        draw.text((60 - offset_t + offset_x, 60 + offset_y), text, fill=color, font=self.font)
        return img

    def _add_card_amount(self, img: Image.Image, amount: int) -> Image.Image:
        img = img.copy()
        quantity_base = Image.open(self.assets_dir / 'quantity.png').convert('RGBA')
        img.paste(quantity_base, (53, 54), mask=quantity_base.split()[3])
        return self._draw_num_text(img, amount, False, (255, 255, 255), 2, 1)

    def _add_icon(self, base: Image.Image, icon_name: str, x: int, y: int) -> Image.Image:
        base = base.copy()
        icon = Image.open(self.assets_dir / icon_name).convert('RGBA')
        base.paste(icon, (x, y), mask=icon.split()[3])
        return base

    def _entry_card_image(self, entry: CardEntry, amount: int = 1, grey: bool = False) -> Image.Image:
        img = self._load_unit_image(entry)
        img = self._add_rarity_frame(img, entry.rarity)
        if grey:
            img = img.convert('L').convert('RGBA')
        if amount > 1 and not grey:
            img = self._add_card_amount(img, amount)
        return img

    def _image_bytes(self, img: Image.Image, fmt: str = 'PNG') -> bytes:
        buf = io.BytesIO()
        save_img = img.convert('RGBA') if fmt.upper() == 'PNG' else img.convert('RGB')
        save_img.save(buf, format=fmt)
        return buf.getvalue()

    def _get_random_cards(self, origin_cards: dict[int, int], row_num: int, col_num: int, amount: int, bonus: bool, super_rare_prob: float, rare_prob: float):
        size = 80
        margin = 7
        margin_offset_x = 6
        margin_offset_y = 6
        cards_amount = []
        extra_bonus = False
        for _ in range(amount):
            a = self._roll_extra_bonus() if bonus else 1
            cards_amount.append(a)
            if a != 1:
                extra_bonus = True
        offset_y = 18 if extra_bonus else 0
        offset_critical_strike = 7 if extra_bonus else 0
        size_x = col_num * size + (col_num + 1) * margin + 2 * margin_offset_x
        size_y = offset_y + row_num * size + (row_num + 1) * margin + 2 * margin_offset_y + offset_critical_strike
        base = Image.new('RGBA', (size_x, size_y), (255, 255, 255, 255))
        background = Image.open(self.assets_dir / 'background.png').convert('RGBA')
        background = background.resize((size_x, size_y - offset_y), Image.Resampling.LANCZOS)
        base.paste(background, (0, offset_y), mask=background.split()[3])
        if extra_bonus:
            base = self._add_icon(base, 'pokecriticalstrike.png', int(size_x / 2) - 71, int(offset_y / 2) - 2)

        card_counter: dict[int, int] = {}
        rarity_counter = {1: [0, 0], 0: [0, 0], -1: [0, 0]}
        card_descs: list[str] = []
        for i in range(amount):
            entry = self.catalog.random_entry(super_rare_prob, rare_prob)
            new_flag = entry.card_id not in origin_cards and entry.card_id not in card_counter
            card_amount = cards_amount[i]
            card_counter[entry.card_id] = card_counter.get(entry.card_id, 0) + card_amount
            new_string = ' 【NEW】' if new_flag else ''
            card_descs.append(f'{RARITY_LABEL[entry.rarity]}「{self.resolver.fromid(entry.chara_id)}」×{card_amount}{new_string}')
            rarity_counter[entry.rarity][0] += 1
            rarity_counter[entry.rarity][1] += 1 if new_flag else 0

            img = self._entry_card_image(entry, amount=card_amount)
            row_index = i // col_num
            col_index = i % col_num
            coor_x = margin + margin_offset_x + col_index * (size + margin)
            coor_y = margin + margin_offset_y + offset_y + offset_critical_strike + row_index * (size + margin)
            base.paste(img, (coor_x, coor_y), mask=img.split()[3])
            if new_flag:
                base = self._add_icon(base, 'new.png', coor_x + size - 27, coor_y - 5)

        if amount > self._cfg('omit_threshold', int):
            card_descs = []
            desc_map = {1: '超稀有卡', 0: '稀有卡', -1: '普通卡'}
            for rarity in (1, 0, -1):
                count, new_count = rarity_counter[rarity]
                if count <= 0:
                    continue
                suffix = f' (【NEW】x{new_count})' if new_count else ''
                card_descs.append(f'【{desc_map[rarity]}】x{count}{suffix}')

        return card_counter, card_descs, self._image_bytes(base)

    def _build_reward_outgoing(self, text: str, image_bytes: bytes | None = None, at_ids: list[str] | None = None) -> Outgoing:
        return Outgoing(text=text, image_bytes=image_bytes, at_ids=at_ids or [])

    def _handle_poke(self, event: AstrMessageEvent) -> Outgoing | None:
        uid = self._sender_id(event)
        scope = self._scope_id(event)
        key = (scope, uid)
        if not self.cooling_time_limiter.check(uid):
            return None
        self.cooling_time_limiter.start_cd(uid)

        if not self.daily_limiter.check(key) and not self.daily_tip_limiter.check(key):
            self.poke_tip_cd_limiter.start_cd(f'{scope}::{uid}')
        if not self.daily_limiter.check(key) and self.poke_tip_cd_limiter.check(f'{scope}::{uid}'):
            self.daily_tip_limiter.increase(key)
            return Outgoing(text='你今天戳得已经够多了，再戳也不会掉卡了~', at_ids=[uid])

        self.daily_tip_limiter.reset(key)
        if (not self.daily_limiter.check(key)) or random.random() > self._cfg('poke_get_cards_probability', float):
            return Outgoing(text='戳', at_ids=[uid])

        amount = self._roll_cards_amount()
        col_num = math.ceil(amount / 2)
        row_num = 2 if amount != 1 else 1
        owned = self.db.get_cards_num(scope, uid)
        card_counter, card_descs, image_bytes = self._get_random_cards(
            origin_cards=owned,
            row_num=row_num,
            col_num=col_num,
            amount=amount,
            bonus=True,
            super_rare_prob=self._cfg('super_rare_probability', float),
            rare_prob=self._cfg('rare_probability', float),
        )
        for card_id, card_amount in card_counter.items():
            self.db.add_card_num(scope, uid, card_id, card_amount)
        self.daily_limiter.increase(key)
        msg = '别戳了别戳了，这些卡送给你了。\n' + '获得了：\n' + '\n'.join(card_descs)
        return Outgoing(text=msg, image_bytes=image_bytes, at_ids=[uid])

    def _handle_mix(self, event: AstrMessageEvent, rest: str) -> Outgoing:
        args = [s for s in rest.split() if s]
        if len(args) != 2:
            return Outgoing(text='请输入想要合成的两张卡，以空格分隔。')
        card1_id = self.catalog.resolve_card_id_by_name(args[0])
        card2_id = self.catalog.resolve_card_id_by_name(args[1])
        if not card1_id:
            return Outgoing(text=f'错误：无法识别 {args[0]}。若为稀有/超稀有卡，请在名称前加上“稀有”或“超稀有”。')
        if not card2_id:
            return Outgoing(text=f'错误：无法识别 {args[1]}。若为稀有/超稀有卡，请在名称前加上“稀有”或“超稀有”。')
        scope = self._scope_id(event)
        uid = self._sender_id(event)
        card1_num = self.db.get_card_num(scope, uid, card1_id)
        card2_num = self.db.get_card_num(scope, uid, card2_id)
        if card1_id == card2_id:
            if card1_num < 2:
                return Outgoing(text=f'{self.catalog.get_card_name_with_rarity(args[0])} 数量不足，无法合成。')
        else:
            if card1_num < 1:
                return Outgoing(text=f'{self.catalog.get_card_name_with_rarity(args[0])} 数量不足，无法合成。')
            if card2_num < 1:
                return Outgoing(text=f'{self.catalog.get_card_name_with_rarity(args[1])} 数量不足，无法合成。')

        rarity1 = self.catalog.get_card_rarity(card1_id)
        rarity2 = self.catalog.get_card_rarity(card2_id)
        _, rare_prob, super_rare_prob = MIX_PROBABILITY[str(sorted([rarity1, rarity2]))]
        owned = self.db.get_cards_num(scope, uid)
        card_counter, _card_descs, image_bytes = self._get_random_cards(owned, 1, 1, 1, False, super_rare_prob, rare_prob)
        new_card_id = next(iter(card_counter.keys()))
        desc, name, _ = self.catalog.get_chara_name(new_card_id)
        self.db.add_card_num(scope, uid, card1_id, -1)
        self.db.add_card_num(scope, uid, card2_id, -1)
        self.db.add_card_num(scope, uid, new_card_id, 1)
        return Outgoing(text=f'融合成功，获得了 {desc}「{name}」×1。', image_bytes=image_bytes, at_ids=[uid])

    def _handle_auto_mix(self, event: AstrMessageEvent, rest: str) -> Outgoing:
        args = [s for s in rest.split() if s]
        if len(args) not in (2, 3):
            return Outgoing(text='参数格式错误。正确格式：一键合成 稀有度1 稀有度2 [轮数]')
        if args[0] not in RARITY_DESC_TO_RARITY or args[1] not in RARITY_DESC_TO_RARITY:
            return Outgoing(text='稀有度仅支持：普通 / 稀有 / 超稀有。')
        if len(args) == 3 and (not args[2].isdigit() or int(args[2]) <= 0):
            return Outgoing(text='合成轮数必须是正整数。')

        scope = self._scope_id(event)
        uid = self._sender_id(event)
        surplus_cards = self.db.get_surplus_cards(scope, uid)
        surplus_cards = {cid: num for cid, num in surplus_cards.items() if cid in self.catalog.entries_by_card_id}
        rarity1 = RARITY_DESC_TO_RARITY[args[0]]
        rarity2 = RARITY_DESC_TO_RARITY[args[1]]

        if args[0] == args[1]:
            available = {cid: num for cid, num in surplus_cards.items() if self.catalog.get_card_rarity(cid) == rarity1}
            total = sum(available.values())
            if len(args) == 3 and int(args[2]) * 2 > total:
                return Outgoing(text=f'合成失败，多余的【{args[0]}】卡数量不足。')
            if len(args) == 2 and total < 2:
                return Outgoing(text=f'合成失败，多余的【{args[0]}】卡数量不足。')
            mix_rounds = int(args[2]) if len(args) == 3 else total // 2
            need = mix_rounds * 2
            consumed = 0
            for cid, num in available.items():
                take = min(num, need - consumed)
                if take <= 0:
                    break
                self.db.add_card_num(scope, uid, cid, -take)
                consumed += take
                if consumed >= need:
                    break
        else:
            available1 = {cid: num for cid, num in surplus_cards.items() if self.catalog.get_card_rarity(cid) == rarity1}
            available2 = {cid: num for cid, num in surplus_cards.items() if self.catalog.get_card_rarity(cid) == rarity2}
            total1 = sum(available1.values())
            total2 = sum(available2.values())
            if len(args) == 3:
                target_rounds = int(args[2])
                if target_rounds > total1 or target_rounds > total2:
                    return Outgoing(text='合成失败，可用多余卡不足。')
                mix_rounds = target_rounds
            else:
                if total1 < 1 or total2 < 1:
                    return Outgoing(text='合成失败，可用多余卡不足。')
                mix_rounds = min(total1, total2)
            for available in (available1, available2):
                consumed = 0
                for cid, num in available.items():
                    take = min(num, mix_rounds - consumed)
                    if take <= 0:
                        break
                    self.db.add_card_num(scope, uid, cid, -take)
                    consumed += take
                    if consumed >= mix_rounds:
                        break

        _, rare_prob, super_rare_prob = OK_MIX_PROBABILITY[str(sorted([rarity1, rarity2]))]
        col_num = max(1, math.ceil(math.sqrt(mix_rounds)))
        row_num = max(1, math.ceil(mix_rounds / col_num))
        owned = self.db.get_cards_num(scope, uid)
        card_counter, card_descs, image_bytes = self._get_random_cards(owned, row_num, col_num, mix_rounds, False, super_rare_prob, rare_prob)
        for card_id, card_amount in card_counter.items():
            self.db.add_card_num(scope, uid, card_id, card_amount)
        text = f'进行了 {mix_rounds} 轮融合，获得了：\n' + '\n'.join(card_descs)
        return Outgoing(text=text, image_bytes=image_bytes, at_ids=[uid])

    def _handle_exchange(self, event: AstrMessageEvent) -> Outgoing:
        segments = self._segments(event)
        at_indexes = [idx for idx, seg in enumerate(segments) if self._is_at_segment(seg)]
        if len(at_indexes) != 1:
            return Outgoing(text='参数格式错误。正确格式：交换 卡1 @某人 卡2')
        at_index = at_indexes[0]
        target_uid = str(getattr(segments[at_index], 'qq', ''))
        if not target_uid:
            return Outgoing(text='无法识别目标用户。')

        before_text = ''.join(getattr(seg, 'text', '') for seg in segments[:at_index] if 'plain' in self._segment_type(seg).lower() or 'text' in self._segment_type(seg).lower())
        after_text = ''.join(getattr(seg, 'text', '') for seg in segments[at_index + 1:] if 'plain' in self._segment_type(seg).lower() or 'text' in self._segment_type(seg).lower())
        card1_name = self._strip_alias_prefix(before_text, EXCHANGE_ALIASES)
        card2_name = after_text.strip()
        if not card1_name or not card2_name:
            return Outgoing(text='参数格式错误。正确格式：交换 卡1 @某人 卡2')

        card1_id = self.catalog.resolve_card_id_by_name(card1_name)
        card2_id = self.catalog.resolve_card_id_by_name(card2_name)
        if not card1_id:
            return Outgoing(text=f'错误：无法识别 {card1_name}。')
        if not card2_id:
            return Outgoing(text=f'错误：无法识别 {card2_name}。')

        scope = self._scope_id(event)
        uid = self._sender_id(event)
        if self.db.get_card_num(scope, uid, card1_id) < 1:
            return Outgoing(text=f'你的 {self.catalog.get_card_name_with_rarity(card1_name)} 数量不足，无法交换。')
        if self.db.get_card_num(scope, target_uid, card2_id) < 1:
            return Outgoing(text=f'对方的 {self.catalog.get_card_name_with_rarity(card2_name)} 数量不足，无法交换。')
        if self.exchange_request_master.has_exchange_request_to_confirm(scope, target_uid):
            return Outgoing(text='该用户当前正有待确认的交易，请稍后再试。')

        req = ExchangeRequest(
            sender_uid=uid,
            card1_id=card1_id,
            card1_name=card1_name,
            target_uid=target_uid,
            card2_id=card2_id,
            card2_name=card2_name,
        )
        self.exchange_request_master.add_exchange_request(scope, target_uid, req)
        valid_time = self._cfg('request_valid_time', int)
        text = (
            f'希望用 {self.catalog.get_card_name_with_rarity(card1_name)} '
            f'交换 {self.catalog.get_card_name_with_rarity(card2_name)}。\n'
            f'请在 {valid_time} 秒内发送“确认交换”。'
        )
        return Outgoing(text=text, at_ids=[target_uid])

    def _handle_confirm_exchange(self, event: AstrMessageEvent) -> Outgoing:
        scope = self._scope_id(event)
        uid = self._sender_id(event)
        req = self.exchange_request_master.get_exchange_request(scope, uid)
        if not req:
            return Outgoing(text='你当前没有待确认的换卡请求。')
        self.exchange_request_master.delete_exchange_request(scope, uid)
        if self.db.get_card_num(scope, req.sender_uid, req.card1_id) < 1:
            return Outgoing(text='发起方卡片数量不足，无法交换。')
        if self.db.get_card_num(scope, req.target_uid, req.card2_id) < 1:
            return Outgoing(text='接收方卡片数量不足，无法交换。')
        self.db.add_card_num(scope, req.sender_uid, req.card1_id, -1)
        self.db.add_card_num(scope, req.target_uid, req.card2_id, -1)
        self.db.add_card_num(scope, req.sender_uid, req.card2_id, 1)
        self.db.add_card_num(scope, req.target_uid, req.card1_id, 1)
        return Outgoing(text='交换成功。')

    def _handle_give(self, event: AstrMessageEvent) -> Outgoing:
        segments = self._segments(event)
        at_indexes = [idx for idx, seg in enumerate(segments) if self._is_at_segment(seg)]
        if len(at_indexes) != 1:
            return Outgoing(text='参数格式错误。正确格式：赠送 @某人 卡名')
        at_index = at_indexes[0]
        target_uid = str(getattr(segments[at_index], 'qq', ''))
        if not target_uid:
            return Outgoing(text='无法识别目标用户。')
        after_text = ''.join(getattr(seg, 'text', '') for seg in segments[at_index + 1:] if 'plain' in self._segment_type(seg).lower() or 'text' in self._segment_type(seg).lower())
        card_name = after_text.strip()
        if not card_name:
            # 兼容纯文本参数
            content = self._normalized_content(event)
            after_alias = self._strip_alias_prefix(content, GIVE_ALIASES)
            if after_alias and ' ' in after_alias:
                card_name = after_alias.split(' ', 1)[1].strip()
        if not card_name:
            return Outgoing(text='参数格式错误。正确格式：赠送 @某人 卡名')

        scope = self._scope_id(event)
        uid = self._sender_id(event)
        if target_uid == uid:
            return Outgoing(text='不用给自己赠卡。')
        if not self.daily_give_limiter.check((scope, target_uid)):
            return Outgoing(text='对方今日接受赠送次数已达上限。')
        card_id = self.catalog.resolve_card_id_by_name(card_name)
        if not card_id:
            return Outgoing(text=f'错误：无法识别 {card_name}。')
        if self.db.get_card_num(scope, uid, card_id) < 1:
            return Outgoing(text=f'{self.catalog.get_card_name_with_rarity(card_name)} 数量不足，无法赠送。')
        self.db.add_card_num(scope, uid, card_id, -1)
        self.db.add_card_num(scope, target_uid, card_id, 1)
        self.daily_give_limiter.increase((scope, target_uid))
        return Outgoing(text=f'已将 {self.catalog.get_card_name_with_rarity(card_name)} 赠送给对方。', at_ids=[uid, target_uid])

    def _handle_storage_compare(self, event: AstrMessageEvent) -> Outgoing:
        at_ids = self._extract_at_ids(event)
        if len(at_ids) != 1:
            return Outgoing(text='参数格式错误。正确格式：仓库对比 @某人')
        target_uid = at_ids[0]
        uid = self._sender_id(event)
        if target_uid == uid:
            return Outgoing(text='不能和自己对比。')
        scope = self._scope_id(event)
        self_cards = self.db.get_cards_num(scope, uid)
        target_cards = self.db.get_cards_num(scope, target_uid)

        i_have_he_not = []
        he_have_i_not = []
        i_have_he_not_more = []
        he_have_i_not_more = []
        for card_id, entry in self.catalog.entries_by_card_id.items():
            name = self.catalog.get_chara_name(card_id)[2]
            in_self = card_id in self_cards
            in_target = card_id in target_cards
            if in_self and not in_target:
                i_have_he_not.append(name)
                if self_cards[card_id] > 1:
                    i_have_he_not_more.append(name)
            if in_target and not in_self:
                he_have_i_not.append(name)
                if target_cards[card_id] > 1:
                    he_have_i_not_more.append(name)

        text = (
            f'你有 TA 没有的：{", ".join(i_have_he_not) or "无"}\n\n'
            f'TA 有你没有的：{", ".join(he_have_i_not) or "无"}\n\n'
            f'你有 TA 没有且你有多余的：{", ".join(i_have_he_not_more) or "无"}\n\n'
            f'TA 有你没有且 TA 有多余的：{", ".join(he_have_i_not_more) or "无"}'
        )
        return Outgoing(text=text)

    def _build_storage_image(self, cards_num: dict[int, int]) -> bytes:
        col_num = max(1, self._cfg('col_num', int))
        rarity_order = [-1, 0, 1]
        row_counts = {r: max(1, math.ceil(len(self.catalog.entries_by_rarity[r]) / col_num)) for r in rarity_order}
        total_rows = sum(row_counts.values())
        width = 40 + col_num * 80 + (col_num - 1) * 10
        height = 120 + total_rows * 80 + (total_rows - 1) * 10
        frame = Image.open(self.assets_dir / 'frame.png').convert('RGBA')
        base = frame.resize((width, height), Image.Resampling.LANCZOS)

        row_index_offset = 0
        row_offset = 0
        for rarity in rarity_order:
            entries = self.catalog.entries_by_rarity[rarity]
            for index, entry in enumerate(entries):
                row_index = index // col_num + row_index_offset
                col_index = index % col_num
                img = self._entry_card_image(entry, amount=cards_num.get(entry.card_id, 1), grey=entry.card_id not in cards_num)
                x = 30 + col_index * 80 + (col_index - 1) * 10
                y = row_offset + 40 + row_index * 80 + (row_index - 1) * 10
                base.paste(img, (x, y), mask=img.split()[3])
            row_index_offset += row_counts[rarity]
            row_offset += 30
        return self._image_bytes(base, fmt='PNG')

    def _handle_storage(self, event: AstrMessageEvent) -> Outgoing:
        at_ids = self._extract_at_ids(event)
        target_uid = at_ids[0] if at_ids else self._sender_id(event)
        scope = self._scope_id(event)
        cards_num = self.db.get_cards_num(scope, target_uid)
        cards_num = {cid: amount for cid, amount in cards_num.items() if cid in self.catalog.entries_by_card_id}
        ranking = self.db.get_scope_ranking(scope, target_uid)
        ranking_desc = f'第 {ranking} 位' if ranking != -1 else '未上榜'
        total_card_num = sum(cards_num.values())
        super_rare_num = len([cid for cid in cards_num if self.catalog.get_card_rarity(cid) == 1])
        rare_num = len([cid for cid in cards_num if self.catalog.get_card_rarity(cid) == 0])
        normal_num = len([cid for cid in cards_num if self.catalog.get_card_rarity(cid) == -1])
        super_total = len(self.catalog.entries_by_rarity[1])
        rare_total = len(self.catalog.entries_by_rarity[0])
        normal_total = len(self.catalog.entries_by_rarity[-1])
        image_bytes = self._build_storage_image(cards_num)
        text = (
            f'仓库统计\n'
            f'持有卡片数：{total_card_num}\n'
            f'普通卡收集：{self._normalize_digit_format(normal_num)}/{self._normalize_digit_format(normal_total)}\n'
            f'稀有卡收集：{self._normalize_digit_format(rare_num)}/{self._normalize_digit_format(rare_total)}\n'
            f'超稀有收集：{self._normalize_digit_format(super_rare_num)}/{self._normalize_digit_format(super_total)}\n'
            f'图鉴完成度：{self._normalize_digit_format(len(cards_num))}/{self._normalize_digit_format(len(self.catalog.entries_by_card_id))}\n'
            f'当前会话排名：{ranking_desc}'
        )
        return Outgoing(text=text, image_bytes=image_bytes, at_ids=[target_uid])

    def _handle_refresh_cards(self) -> Outgoing:
        stats = self.catalog.refresh()
        text = (
            f'卡池已刷新。普通={stats["normal"]}，稀有={stats["rare"]}，超稀有={stats["super_rare"]}，'
            f'真实头像={stats["real_assets"]}'
        )
        return Outgoing(text=text)


@register('astrbot_plugin_pokemanpcr', 'satori5555', '将 poke-man-pcr 迁移到 AstrBot 的公主连结卡片插件', '1.0.0', 'https://github.com/satori5555/astrbot_plugin_pokemanpcr')
class Main(PokeManPCRPlugin):
    pass
