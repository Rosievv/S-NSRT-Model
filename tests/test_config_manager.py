"""
Unit tests for ConfigManager secret redaction
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config_manager import ConfigManager


def _bare_config_manager(config_dict):
    """Build a standalone ConfigManager instance without touching the
    process-wide singleton (bypasses ConfigManager.__new__ on purpose)."""
    cm = object.__new__(ConfigManager)
    cm._config = config_dict
    return cm


def test_get_redacted_masks_api_keys():
    """Direct nested api_key values should be masked."""
    cm = _bare_config_manager({
        'data_sources': {
            'us_census': {'api_key': 'super-secret-census-key', 'base_url': 'https://example.com'},
            'fred': {'api_key': 'super-secret-fred-key'},
        }
    })

    redacted = cm.get_redacted()

    assert redacted['data_sources']['us_census']['api_key'] == '***REDACTED***'
    assert redacted['data_sources']['fred']['api_key'] == '***REDACTED***'
    # Non-secret values are left untouched
    assert redacted['data_sources']['us_census']['base_url'] == 'https://example.com'


def test_get_redacted_masks_tokens_secrets_passwords():
    """Other common secret-like suffixes should also be masked."""
    cm = _bare_config_manager({
        'service': {
            'auth_token': 'abc123',
            'client_secret': 'shh',
            'db_password': 'hunter2',
            'retry_attempts': 3,
        }
    })

    redacted = cm.get_redacted()

    assert redacted['service']['auth_token'] == '***REDACTED***'
    assert redacted['service']['client_secret'] == '***REDACTED***'
    assert redacted['service']['db_password'] == '***REDACTED***'
    # Non-secret values (including falsy-looking but non-secret numbers) untouched
    assert redacted['service']['retry_attempts'] == 3


def test_get_redacted_does_not_mutate_original_config():
    """get_redacted() must return a copy, not mutate the live config."""
    cm = _bare_config_manager({'data_sources': {'us_census': {'api_key': 'super-secret'}}})

    cm.get_redacted()

    assert cm._config['data_sources']['us_census']['api_key'] == 'super-secret'


def test_repr_never_leaks_secret_values():
    """__repr__ should be safe to log/print directly."""
    cm = _bare_config_manager({'data_sources': {'us_census': {'api_key': 'super-secret-value'}}})

    assert 'super-secret-value' not in repr(cm)
    assert '***REDACTED***' in repr(cm)

