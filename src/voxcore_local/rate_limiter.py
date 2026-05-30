"""
Rate limiter + retry pour les appels externes (LLM, TTS).

- Limite le nombre de requêtes par seconde
- Retries avec backoff exponentiel
- Graceful degradation quand un service est indisponible
"""
import time
import threading
from typing import Optional, Callable, Any
from functools import wraps


class RateLimiter:
    """
    Rate limiter token bucket.
    
    Permet N requêtes par seconde avec un burst max.
    """
    
    def __init__(self, rate: float = 10.0, burst: int = 20):
        """
        Args:
            rate: Nombre de requêtes autorisées par seconde
            burst: Burst maximum (tokens initiaux)
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: float = 1.0) -> float:
        """
        Acquitte des tokens. Retourne le délai d'attente (0 = immédiat).
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            else:
                deficit = tokens - self._tokens
                wait = deficit / self.rate
                return wait
    
    def wait(self, tokens: float = 1.0):
        """Bloque jusqu'à ce que les tokens soient disponibles."""
        delay = self.acquire(tokens)
        if delay > 0:
            time.sleep(delay)


class Retry:
    """
    Retry avec backoff exponentiel.
    
    Usage :
        retry = Retry(max_attempts=3, base_delay=0.5)
        result = retry.call(my_function, arg1, arg2)
    """
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 0.5,
                 max_delay: float = 10.0, exceptions: tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
    
    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Exécute la fonction avec retry."""
        last_error = None
        for attempt in range(self.max_attempts):
            try:
                return fn(*args, **kwargs)
            except self.exceptions as e:
                last_error = e
                if attempt < self.max_attempts - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
        raise last_error
    
    def decorator(self) -> Callable:
        """Décorateur pour wrap une fonction existante."""
        def decorate(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.call(fn, *args, **kwargs)
            return wrapper
        return decorate


def with_retry(max_attempts: int = 3, base_delay: float = 0.5,
               max_delay: float = 10.0, exceptions: tuple = (Exception,)):
    """
    Décorateur shortcut pour retry.
    
    Usage :
        @with_retry(max_attempts=3)
        def call_api():
            ...
    """
    retry = Retry(max_attempts=max_attempts, base_delay=base_delay,
                  max_delay=max_delay, exceptions=exceptions)
    return retry.decorator()


def with_rate_limit(rate: float = 10.0, burst: int = 20):
    """
    Décorateur shortcut pour rate limiting.
    
    Usage :
        limiter = RateLimiter(rate=5)  # 5 req/s
        
        @with_rate_limit(limiter=limiter)
        def call_api():
            ...
    """
    def decorate(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Rate limiter passé en paramètre ou créé
            rl = kwargs.pop('_limiter', None) or getattr(fn, '_limiter', None)
            if rl:
                rl.wait()
            return fn(*args, **kwargs)
        return wrapper
    return decorate


class CircuitBreaker:
    """
    Circuit breaker — stoppe les appels quand un service échoue trop souvent.
    
    Trois états :
    - CLOSED: normal, appels autorisés
    - OPEN: circuit ouvert, appels rejetés immédiatement
    - HALF-OPEN: test en cours, un appel autorisé
    
    Usage :
        cb = CircuitBreaker(failure_threshold=5, recovery_time=30)
        result = cb.call(risky_function)
    """
    
    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"
    
    def __init__(self, failure_threshold: int = 5, recovery_time: float = 30.0,
                 exceptions: tuple = (Exception,)):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.exceptions = exceptions
        
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._last_failure = 0.0
        self._lock = threading.Lock()
    
    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self._state == self.STATE_OPEN:
                if time.monotonic() - self._last_failure >= self.recovery_time:
                    self._state = self.STATE_HALF_OPEN
                else:
                    raise RuntimeError(f"Circuit breaker OPEN — service indisponible")
        
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self._failure_count = 0
                self._state = self.STATE_CLOSED
            return result
        except self.exceptions as e:
            with self._lock:
                self._failure_count += 1
                self._last_failure = time.monotonic()
                if self._failure_count >= self.failure_threshold:
                    self._state = self.STATE_OPEN
            raise e
    
    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.STATE_OPEN:
                if time.monotonic() - self._last_failure >= self.recovery_time:
                    self._state = self.STATE_HALF_OPEN
            return self._state
    
    def reset(self):
        """Réinitialise le circuit breaker."""
        with self._lock:
            self._state = self.STATE_CLOSED
            self._failure_count = 0
