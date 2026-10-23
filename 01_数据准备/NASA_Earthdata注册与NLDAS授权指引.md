# NASA Earthdata 注册与 NLDAS 授权指引

核验日期：2026-10-23。适用于 ECHO 原网格气象资料准备；账号注册和邮箱激活由账号本人完成。

## 1. 注册免费账号

打开官方注册页：<https://urs.earthdata.nasa.gov/users/new>。

- Username：可尝试 `dingni1226`，如已被占用再调整。用户名为 4—30 个字符，至少含一个字母，只允许小写字母、数字、英文句点和下划线；不能用 `ding-ni` 中的连字符，也不能有空格、首尾特殊符号或连续特殊符号。
- Password：当前页面要求至少 12 个字符，并包含大写字母、小写字母、数字和特殊字符。请自行设置并保存在自己的密码管理器中。
- First Name / Last Name：分别填写本人的名和姓；邮箱填写能够接收邮件的地址。
- Country：按实际情况填写。
- Affiliation：如果目前在高校学习，可选 `Education`；Organization 填实际学校或机构。
- Study Area：本项目可选 `Hydrologic Cycle`。
- User Type：普通资料使用者可选 `Public User`；其余资料按实际情况填写。

提交前以页面当前显示的必填项和规则为准。Earthdata 普通账号注册免费，不需要购买云服务。

## 2. 激活邮箱并登录

提交注册后，在邮箱中查找 Earthdata 的激活邮件，点击其中的激活链接；未收到时检查垃圾邮件。官方说明要求激活后才能登录。

回到 <https://urs.earthdata.nasa.gov/>，确认可以登录。

## 3. 授权 GES DISC 数据应用

优先直接访问本项目的 NLDAS 小样例：

<https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/2011/274/NLDAS_FORA0125_H.A20111001.0000.020.nc>

按提示使用 Earthdata 登录；首次使用出现应用授权或使用协议时，核对为此次 NASA GES DISC 数据服务，再阅读并完成授权。下载一个样例用于确认访问权限即可，不必手工下载多年资料。

如命令行访问仍提示未授权，可按官方预授权步骤操作：

1. 登录 Earthdata，进入 `Applications` → `Authorized Apps`。
2. 点击 `APPROVE MORE APPLICATIONS`。
3. 搜索 `GES DISC`，核对相应 NASA GES DISC 数据访问应用，完成 `AUTHORIZE` 及页面要求的协议。
4. 回到样例链接再次检查。已经授权的应用不会出现在待授权搜索结果中，可先查看已授权列表。

官方页面实际显示的应用名和选项为准，不填写或猜测第三方应用的 client ID。

## 4. 告知项目继续准备下载

完成后只需在聊天里说“已注册并激活”；如果样例也已下载，请补充“样例能下载”。不要发送密码、访问令牌、邮箱验证码或激活链接。

浏览器登录成功不等于 Python 或下载器已经认证。后续需要单独配置本机下载认证，再验证一个小文件的状态、格式、变量、单位和时间语义，随后生成批量下载清单。凭据不得进入项目代码、配置快照、日志或 Git；不得把未认证返回的 HTML 登录页当作 NetCDF 数据。

## 5. 生成 Bearer 令牌

按 NASA 官方《User Token Management》（页面更新日期 2026-07-01，本地于 2026-10-23 核验）：

1. 在自己的浏览器登录 <https://urs.earthdata.nasa.gov/>。
2. 在个人资料页面的导航中点击 `Generate Token`。
3. 在令牌生成页面底部点击 `Generate Token` 按钮。
4. 新令牌默认隐藏；点击 `Show Token` 后自行保存到个人密码管理器，不要粘贴到聊天、项目说明或 Git。

官方当前规定：令牌有效期为 60 天，每个账号最多同时持有两个有效令牌。如已经存在可用令牌，可直接复用，不必重复生成；不要为了本项目擅自撤销其他正在使用的令牌。

用户名、姓名和邮箱不足以代为生成令牌，需要已经认证的会话或账号凭据。项目不收集聊天中的密码。令牌生成后还需单独配置下载器认证并测试 GES DISC 对该认证方式的支持；不能仅凭令牌存在就宣称 NLDAS 下载已打通。

## 6. 对 ECHO 数据与模型的作用

本次申请用于原网格 `NLDAS_FORA0125_H` v`2.0`，以真实格点气象按节点区域加权，保留降雨空间位置。它是完整空间模型的数据准备步骤。已有流域平均气象仍可用于基线和相应核查，但不能复制到不同节点后宣称已经保留空间降雨信息。

USGS DEM 下载不依赖 Earthdata 账号。账号开通也不代表完整主模型训练已经完成；仍需完成真实空间资料核验、阶段训练编排和官方 Mamba-3 Linux GPU 验证。

## 官方来源

- 注册页：<https://urs.earthdata.nasa.gov/users/new>
- 注册及邮件激活说明：<https://urs.earthdata.nasa.gov/documentation/for_users/how_to_register>
- 应用授权说明：<https://urs.earthdata.nasa.gov/documentation/for_users/what_is_app_auth>
- 预授权操作说明：<https://urs.earthdata.nasa.gov/documentation/for_users/how_to_preauth_app>
- 令牌生成与管理：<https://urs.earthdata.nasa.gov/documentation/for_users/user_token>
- NLDAS 产品页：<https://disc.gsfc.nasa.gov/datacollection/NLDAS_FORA0125_H_2.0.html>

本次官方注册页和授权文档核验快照保存在本地忽略目录 `本地参考资料/sources/earthdata_registration_20261023/`。GES DISC 介绍页为动态页面，本次静态请求只取得页面外壳；授权步骤依据上列实际读取成功的 Earthdata 官方文档。
