from collections.abc import Callable

from backend.models.anthropic import AnthropicModelList
from backend.models.gemini import GeminiModelList
from backend.models.local import Context_type, QchatModelInfo, QchatModelList, CurrentLocalConfig
from backend.models.openai import OpenAIModelList


class ModelListConverter:
    @classmethod
    def to_qchat_model_list(
        cls,
        model_list: OpenAIModelList | GeminiModelList | AnthropicModelList,
        support_context_resolver: Callable[[str], list[Context_type]] | None = None,
    ) -> QchatModelList:
        if support_context_resolver is None:
            resolver: Callable[[str], list[Context_type]] = cls._default_context_resolver
        else:
            resolver = support_context_resolver
        match model_list:
            case OpenAIModelList():
                return cls.from_openai(model_list, resolver)
            case GeminiModelList():
                return cls.from_gemini(model_list, resolver)
            case AnthropicModelList():
                return cls.from_anthropic(model_list, resolver)
            case _:
                raise ValueError(f"Unsupported model list type: {type(model_list)}")

    @staticmethod
    def _default_context_resolver(_name: str) -> list[Context_type]:
        return ["text"]

    @classmethod
    def from_openai(
        cls,
        model_list: OpenAIModelList,
        support_context_resolver: Callable[[str], list[Context_type]],
    ) -> QchatModelList:
        return QchatModelList(
            model_list=[
                QchatModelInfo(
                    model_name=model.id,
                    support_context=support_context_resolver(model.id),
                )
                for model in model_list.data
            ]
        )

    @classmethod
    def from_gemini(
        cls,
        model_list: GeminiModelList,
        support_context_resolver: Callable[[str], list[Context_type]],
    ) -> QchatModelList:
        return QchatModelList(
            model_list=[
                QchatModelInfo(
                    model_name=cls._normalize_gemini_name(model.name),
                    support_context=support_context_resolver(
                        cls._normalize_gemini_name(model.name)
                    ),
                )
                for model in model_list.models
            ]
        )

    @classmethod
    def from_anthropic(
        cls,
        model_list: AnthropicModelList,
        support_context_resolver: Callable[[str], list[Context_type]],
    ) -> QchatModelList:
        return QchatModelList(
            model_list=[
                QchatModelInfo(
                    model_name=model.id,
                    support_context=support_context_resolver(model.id),
                )
                for model in model_list.data
            ]
        )

    @staticmethod
    def _normalize_gemini_name(name: str) -> str:
        if name.startswith("models/"):
            return name[len("models/") :]
        return name

