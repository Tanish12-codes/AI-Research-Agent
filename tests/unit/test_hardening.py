"""Regression tests for SSRF hardening, URL/repo parsing and pros/cons matching."""
from datetime import datetime, timezone

import httpx
import pytest

from research_agent.confidence import _apply_question_type_rules
from research_agent.schemas import Evidence, QuestionType
from research_agent.security import resolve_public_host, sanitize_output_urls, validate_url
from research_agent.tools.fetch import fetch_page
from research_agent.tools.github import extract_github_repos


async def _allow_all(_host):
    return True, None


class TestUrlValidation:
    @pytest.mark.parametrize('url', [
        'http://2130706433/',
        'http://0x7f000001/',
        'http://127.1/',
        'http://0177.0.0.1/',
        'http://[::ffff:127.0.0.1]/',
        'http://0.0.0.0/',
        'http://app.localhost/',
    ])
    def test_non_standard_private_forms_blocked(self, url):
        valid, reason = validate_url(url)
        assert valid is False, reason

    def test_public_ip_allowed(self):
        assert validate_url('https://93.184.216.34/')[0] is True

    @pytest.mark.asyncio
    async def test_dns_name_resolving_to_loopback_blocked(self):
        ok, reason = await resolve_public_host('localhost')
        assert ok is False
        assert 'non-public' in reason


class TestFetchPage:
    @pytest.mark.asyncio
    async def test_streamed_body_over_cap_rejected(self):
        def handler(request):
            return httpx.Response(200, headers={'content-type': 'text/html'}, content=iter([b'x' * 600] * 4))

        result = await fetch_page('https://example.com/big', max_size=1000,
                                  resolver=_allow_all, transport=httpx.MockTransport(handler))
        assert result is None

    @pytest.mark.asyncio
    async def test_redirect_to_private_dns_blocked(self):
        seen = []

        async def resolver(host):
            seen.append(host)
            return (host != 'rebind.example'), 'private'

        def handler(request):
            if request.url.host == 'example.com':
                return httpx.Response(302, headers={'location': 'https://rebind.example/meta'})
            return httpx.Response(200, headers={'content-type': 'text/html'}, text='<p>secret</p>')

        result = await fetch_page('https://example.com/start', resolver=resolver,
                                  transport=httpx.MockTransport(handler))
        assert result is None
        assert seen == ['example.com', 'rebind.example']

    @pytest.mark.asyncio
    async def test_normal_page_fetched(self):
        def handler(request):
            return httpx.Response(200, headers={'content-type': 'text/html'},
                                  text='<html><title>Doc</title><p>Python is a language.</p></html>')

        result = await fetch_page('https://example.com/doc', resolver=_allow_all,
                                  transport=httpx.MockTransport(handler))
        assert result is not None
        assert result.title == 'Doc'
        assert 'Python is a language.' in result.text


class TestParsingFixes:
    def test_output_sanitizer_keeps_domains_starting_with_w(self):
        out = sanitize_output_urls('see https://web.dev/x and https://www.web.dev/y', {'web.dev'})
        assert 'https://web.dev/x' in out
        assert 'https://www.web.dev/y' in out

    def test_github_repo_names_not_mangled(self):
        repos = extract_github_repos([
            'https://github.com/foo/digit',
            'https://github.com/foo/tooling.git',
            'https://github.com/topics/python',
            'https://github.com/orgs/foo/people',
            'https://notgithub.com/a/b',
        ])
        assert repos == [('foo', 'digit'), ('foo', 'tooling')]

    def test_pros_cons_requires_whole_words(self):
        ev = Evidence(evidence_id='e', url='https://example.com', domain='example.com', title='t',
                      text='This process provides content for the console.',
                      retrieval_timestamp=datetime.now(timezone.utc), provider='test')
        cap, _ = _apply_question_type_rules(QuestionType.PROS_CONS, [], [ev], 1.0, [])
        assert cap == 0.5

        ev.text = 'Benefits include speed; drawbacks include complexity.'
        cap, _ = _apply_question_type_rules(QuestionType.PROS_CONS, [], [ev], 1.0, [])
        assert cap == 1.0
